import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive
import yaml
import time
from collections import deque
from scipy.spatial.transform import Rotation as R


class ArucoURController:
    def __init__(self, robot_ip="10.2.30.60"):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.T_base_to_monde = np.eye(4)
        
        try:
            with open('/home/robot/Bureau/Stage/UR5_projetFinale/chemin_QRcode/intrinsics_v2.yaml', 'r') as f:
                intrinsic_data = yaml.safe_load(f)
                self.camera_matrix = np.array(intrinsic_data['camera_matrix'], dtype=np.float32)
                self.dist_coeffs = np.array(intrinsic_data['dist_coeffs'], dtype=np.float32)
            print("Paramètres intrinsèques chargés avec succès")
            
            with open('/home/robot/Bureau/Stage/UR5_projetFinale/chemin_QRcode/hand_eye.yaml', 'r') as f:
                hand_eye_data = yaml.safe_load(f)
                self.T_cam_to_tcp = np.array(hand_eye_data['T_cam_to_flange'], dtype=np.float32)
                R =[[0,1,0],
                    [-1,0,0],
                    [0,0,1]]
                self.T_cam_to_tcp[:3,:3] =self.T_cam_to_tcp[:3,:3] @R
            print("Transformation hand-eye chargée avec succès")
        except Exception as e:
            print(f"Erreur lors du chargement des calibrations: {e}")
            raise
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size = 0.02 
        print("ici")
        
        try:
            self.connexion()
            
        except Exception as e:
            raise
        
        self.update_cam_to_base_transform()
        self.pose_history = {}  
        self.history_length = 5  
        self.last_frame_markers = None 
        self.marker_weights = {} 

        self.last_successful_movement = None 
        self.max_allowed_deviation = 0.05  
        self.scale_correction = 1.0 
    
    def connexion(self):

        self.ur5_receive = rtde_receive.RTDEReceiveInterface("10.2.30.60")

        try:
            self.ur5_control = rtde_control.RTDEControlInterface("10.2.30.60")

        except Exception as e:
            print("Erreur pendant la connexion RTDEControl :", e)

    def deconnexion(self): 
        self.ur5_control.disconnect()
    
    def update_cam_to_base_transform(self):
        tcp_pose = self.ur5_receive.getActualTCPPose()
        if tcp_pose is None or len(tcp_pose) < 6:
            raise ValueError("Erreur : la pose TCP reçue est invalide ou vide.")
        T_tcp_to_base = np.eye(4)
        T_tcp_to_base[:3, 3] = tcp_pose[:3]  
        rx, ry, rz = tcp_pose[3:6]
        rot_vector = np.array([rx, ry, rz])
        rot_matrix, _ = cv2.Rodrigues(rot_vector)
        T_tcp_to_base[:3, :3] = rot_matrix
        self.T_cam_to_base = T_tcp_to_base @ self.T_cam_to_tcp


    def detect_aruco_markers(self):
        """Détecte les marqueurs ArUco et calcule leurs poses avec une meilleure précision"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            print("Images non disponibles")
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        corners, ids, _ = self.detector.detectMarkers(enhanced_gray)
        
        if ids is None or len(ids) == 0:
           
            if self.last_frame_markers is not None:
                
                return color_image, self.last_frame_markers
            return color_image, None
        
        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        
        markers_data = []
        
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            marker_points = np.array([
                [-self.marker_size/2, self.marker_size/2, 0],
                [self.marker_size/2, self.marker_size/2, 0],
                [self.marker_size/2, -self.marker_size/2, 0],
                [-self.marker_size/2, -self.marker_size/2, 0]
            ], dtype=np.float32)
            
            ret, rvec, tvec = cv2.solvePnP(
                marker_points, corner[0],
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            ret, rvec, tvec = cv2.solvePnP(
                marker_points, corner[0],
                self.camera_matrix, self.dist_coeffs,
                rvec=rvec, tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs,
                             rvec, tvec, self.marker_size/2)
            rot_matrix_cam = np.eye(4)
            rot_matrix, _ = cv2.Rodrigues(rvec)
            
            rot_matrix_cam [:3,:3]=rot_matrix
            position = tvec.flatten()
            position=position.tolist()
            for i in range (3):
                rot_matrix_cam[i][3]=position[i]

            center_x = int(np.mean([p[0] for p in corner[0]]))
            center_y = int(np.mean([p[1] for p in corner[0]]))
            
            depth_region = depth_image[
                max(0, center_y-5):min(depth_image.shape[0], center_y+5),
                max(0, center_x-5):min(depth_image.shape[1], center_x+5)
            ]
            valid_depths = depth_region[depth_region > 0]
            
            if len(valid_depths) > 0:
                depth_sensor = np.median(valid_depths) / 1000.0
                
                depth_solvepnp = position[2]
                depth_diff = abs(depth_sensor - depth_solvepnp)
                
                if depth_diff > 0.05: 
                    alpha = 0.7
                    corrected_depth = alpha * depth_sensor + (1 - alpha) * depth_solvepnp
                    
                    position[2] = corrected_depth
                    
                    tvec[2] = corrected_depth
            
            x_axis = rot_matrix[:, 0]
            y_axis = rot_matrix[:, 1]
            z_axis = rot_matrix[:, 2]
            x_axis_cam = [1.0,0.0,0.0]
            cos_theta = np.clip(np.dot(x_axis, x_axis_cam), -1.0, 1.0)
            sin_theta = np.cross(x_axis, x_axis_cam)[2]
            sin_theta = np.clip(sin_theta, -1.0, 1.0)
            angle_rad_x = np.arctan2(sin_theta, cos_theta)

            rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
            rpy = self.rotvec_to_rpy(rvec[0][0], rvec[1][0], rvec[2][0])
            

            pos_hom = np.ones(4)
            pos_hom[:3] = position
            position_base = (self.T_cam_to_base @ pos_hom)[:3]

            R_cam_to_base = self.T_cam_to_base[:3, :3]
            x_axis_base = R_cam_to_base @ x_axis
            y_axis_base = R_cam_to_base @ y_axis
            z_axis_base = R_cam_to_base @ z_axis
            R_aruco_in_base = R_cam_to_base @ rot_matrix
            #print(f"la matrice de l'aruco dans la base est : \n{R_aruco_in_base}"
            rotvec_base = R.from_matrix(R_aruco_in_base).as_rotvec() 
            #print(f"le rotvec de l'aruco dans la base est : \n{rotvec_base}")
            #print("Axes ArUco dans le repère base :")
            #print("X =", x_axis_base)
            #print("Y =", y_axis_base)
            #print("Z =", z_axis_base)

            #print("Axes Caméra dans base :")
            #print("R_cam_to_base =", R_cam_to_base)
            #print("Matrice rotation ArUco :\n", rot_matrix)
            rot_z_90 = R.from_euler('z', np.pi/2).as_matrix()
            diff = rot_matrix @ rot_z_90.T
            #print("Diff avec rot_z_90 :\n", diff)


            if marker_id not in self.pose_history:
                self.pose_history[marker_id] = deque(maxlen=self.history_length)
            
            marker_data = {
                'id': marker_id,
                'position': position_base,
                'pos_marker': pos_hom[:3],
                'angle_rad_x' : angle_rad_x,
                'rvec' : rotvec_base,
                'rpy' :rpy,
                'axes': {
                    'x': x_axis_base,
                    'y': y_axis_base,
                    'z': z_axis_base
                },
                'timestamp': time.time()
            }
            
            self.pose_history[marker_id].append(marker_data)
            
            if len(self.pose_history[marker_id]) > 1:
                filtered_position = np.mean([m['position'] for m in self.pose_history[marker_id]], axis=0)
                filtered_x = np.mean([m['axes']['x'] for m in self.pose_history[marker_id]], axis=0)
                filtered_y = np.mean([m['axes']['y'] for m in self.pose_history[marker_id]], axis=0)
                filtered_z = np.mean([m['axes']['z'] for m in self.pose_history[marker_id]], axis=0)
                
                filtered_x = filtered_x / np.linalg.norm(filtered_x)
                filtered_y = filtered_y / np.linalg.norm(filtered_y)
                filtered_z = filtered_z / np.linalg.norm(filtered_z)
                
                marker_data['position'] = filtered_position
                marker_data['axes']['x'] = filtered_x
                marker_data['axes']['y'] = filtered_y
                marker_data['axes']['z'] = filtered_z
            
            markers_data.append(marker_data)
            
            cv2.putText(color_image, f"ID={marker_id}", (int(corner[0][0][0]), int(corner[0][0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        self.last_frame_markers = markers_data
        
        return color_image, markers_data
    
    def compute_weighted_aruco_reference_frames(self, markers_data):
        """
        Calcule un repère de référence à partir des marqueurs détectés
        en tenant compte de leur fiabilité relative
        """
        if not markers_data or len(markers_data) == 0:
            return None
            
        total_weight = 0
        weighted_positions = np.zeros(3)
        weighted_x_axes = np.zeros(3)
        weighted_y_axes = np.zeros(3)
        weighted_z_axes = np.zeros(3)
        
        all_markers_axes = {}
        
        for marker in markers_data:
            marker_id = marker['id']
            marker_pos = marker['pos_marker']
            all_markers_axes[marker_id] = {
                'position': marker['position'],
                'axes': marker['axes'],
                'pos_marker': marker_pos,
                'angle_rad_x' : marker['angle_rad_x'],
                'rvec' : marker['rvec'],
                'rpy' :marker['rpy']
            }
            
            weight = self.marker_weights.get(marker_id, 1.0)
            
            distance = np.linalg.norm(marker['position'])
            distance_factor = np.exp(-distance / 0.5)  
            weight *= distance_factor
            
            weighted_positions += weight * marker['position']
            weighted_x_axes += weight * marker['axes']['x']
            weighted_y_axes += weight * marker['axes']['y']
            weighted_z_axes += weight * marker['axes']['z']
            total_weight += weight
        
        if total_weight > 0:
            mean_position = weighted_positions / total_weight
            mean_x = weighted_x_axes / total_weight
            mean_y = weighted_y_axes / total_weight
            mean_z = weighted_z_axes / total_weight
        else:
            positions = [marker['position'] for marker in markers_data]
            x_axes = [marker['axes']['x'] for marker in markers_data]
            y_axes = [marker['axes']['y'] for marker in markers_data]
            z_axes = [marker['axes']['z'] for marker in markers_data]
            mean_position = np.mean(positions, axis=0)
            mean_x = np.mean(x_axes, axis=0)
            mean_y = np.mean(y_axes, axis=0)
            mean_z = np.mean(z_axes, axis=0)
        
        mean_z = mean_z / np.linalg.norm(mean_z) 
        mean_y = mean_y - np.dot(mean_y, mean_z) * mean_z
        mean_y = mean_y / np.linalg.norm(mean_y) 
        
        mean_x = np.cross(mean_y, mean_z)
        rot_matrices = [R.from_rotvec(marker['rvec'].flatten()).as_matrix() for marker in markers_data]
        R_avg = sum(rot_matrices) / len(rot_matrices)
        U, _, Vt = np.linalg.svd(R_avg)
        R_orthonorm = U @ Vt
        rvec_avg = R.from_matrix(R_orthonorm).as_rotvec()
        rot_z_180 = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        angle = [marker['angle_rad_x'] for marker in markers_data]
        angle_avg = sum(angle) / len(angle)
        rotated_x = rot_z_180 @ mean_x
        rotated_y = rot_z_180 @ mean_y
        
        if markers_data:
            marker_id = markers_data[0]['id']
            marker_pos = markers_data[0]['pos_marker']
        else:
            marker_id = None
            marker_pos = None
            
        return {
            'id': marker_id,
            'pos_marker': marker_pos,
            'position': mean_position,
            'axes': {
                'x': rotated_x,
                'y': rotated_y,
                'z': mean_z
            },
            'original_axes': {
                'x': mean_x,
                'y': mean_y,
                'z': mean_z
            },
            'rvec_avg' : rvec_avg,
            'angle_avg' : angle_avg,
            'all_markers': all_markers_axes  
        }

    def rpy_to_rotvec(self,roll, pitch, yaw):
        rot = R.from_euler('xyz', [roll, pitch, yaw])
        rotvec = rot.as_rotvec()
        return rotvec

    def rotvec_to_rpy(self,rx, ry, rz):
        rot = R.from_rotvec([rx, ry, rz])
        rpy = rot.as_euler('xyz')
        return rpy
    
    def run_demo(self):
        while True:
            frame, markers_data = self.detect_aruco_markers()

            if frame is not None:
                if markers_data:
                    aruco_frame = self.compute_weighted_aruco_reference_frames(markers_data)
                    if aruco_frame:
                        markers_all_data = aruco_frame['all_markers']
                        marker_ids = list(markers_all_data.keys())

            cv2.imshow("ArUco Markers", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            elif key == ord('c'):
                self.update_cam_to_base_transform()
                _, markers_data = self.detect_aruco_markers()
                self.pose_history.clear()

                if not markers_data:
                    continue

                aruco_frame = self.compute_weighted_aruco_reference_frames(markers_data)
                if not aruco_frame:
                    print("Impossible de calculer le repère ArUco.")
                    continue

                # Obtenir la rotation moyenne (dans le repère cam)
                rvec_avg = aruco_frame['rvec_avg']
                print(f"la valeur de rvec_avg est : {rvec_avg}")
                rpy = self.rotvec_to_rpy(rvec_avg[0],rvec_avg[1],rvec_avg[2])
                rotvec_1 = self.rpy_to_rotvec (np.pi, 0 ,rpy[2] +np.pi )
                
                pose6d = self.ur5_receive.getActualTCPPose()
                for i in range(3) :
                    pose6d[i+3] = rotvec_1[i]
                print(pose6d)
                self.ur5_control.moveL(pose6d,1,1)
                input('le premier mouvement est fini')
                self.update_cam_to_base_transform()
                _, markers_data = self.detect_aruco_markers()
                self.pose_history.clear()

                angle_avg = aruco_frame['angle_avg']
                pose6d = self.ur5_receive.getActualQ()
                pose6d[5]=rpy[2]
                print(angle_avg)
                #self.ur5_control.moveJ(pose6d,1,1)
                # Obtenir la position moyenne en base
                base_pos = [data['position'] for data in aruco_frame['all_markers'].values()]
                mean_pos = np.mean(np.stack(base_pos, axis=0), axis=0)
                offset = np.array([0.035, 0.01, 0.01])
                target_pos = mean_pos + offset

                # Construire la pose finale
                pose6d = self.ur5_receive.getActualTCPPose()

                for i in range(3):
                    pose6d[i] = target_pos[i]

                print(f"Commande de pose: position={target_pos}")
                #self.ur5_control.moveL(pose6d, 0.2, 0.2)
                time.sleep(4)

            elif key == ord('p') :
                frame, markers_data = self.detect_aruco_markers()

                if frame is not None:
                    if markers_data:
                        aruco_frame = self.compute_weighted_aruco_reference_frames(markers_data)
                        if aruco_frame:
                            markers_all_data = aruco_frame['all_markers'] 
                            marker_ids = list(markers_all_data.keys())
                            if 47 in marker_ids :
                                rvec = markers_all_data[47]['rvec']
                                T_aruco_to_cam = cv2.Rodrigues(rvec)
                                T_cam_to_aruco = np.linalg.inv(T_aruco_to_cam)
                                #T_base_to_aruco = T_cam_to_aruco @ T_cam_to_base

        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.ur5_control.disconnect()


if __name__ == "__main__":
    controller = ArucoURController(robot_ip="10.2.30.60")
    controller.run_demo()
