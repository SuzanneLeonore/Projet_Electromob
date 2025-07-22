"""
pick_and_place_system.py - Capture et traite les image pour la calibration. 
Renvoie deux fichiers yaml : hand_eye_data.yaml et intrinsics.yaml

Auteur : Alban CASELLA et Suzanne-Léonore GIRARD-JOLLET
Date : Juin 2025
Description : Ce script utilise la caméra pour capturer des images d'une grille d'ArUco pour calibrer la caméra et déterminer les paramètres intrasects de la caméra.
"""
import cv2
import numpy as np
import yaml
import glob
import pyrealsense2 as rs

# Hand-eye calibration and camera intrinsic calibration utilities
def calibrate_camera_intrinsics(image_files, charuco_board, aruco_dict, output_yaml):
    all_corners = []
    all_ids = []
    image_size = None
    for fname in glob.glob(image_files):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, charuco_board)
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                image_size = gray.shape[::-1]
    # calibrate
    ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, charuco_board, image_size, None, None)
    # save
    with open(output_yaml, 'w') as f:
        yaml.dump({'camera_matrix': camera_matrix.tolist(),
                    'dist_coeffs': dist_coeffs.tolist()}, f)
    print(f"Intrinsics saved to {output_yaml}")

    print(f"Erreur de reprojection : {ret:.4f}")

    return camera_matrix, dist_coeffs


def calibrate_hand_eye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, output_yaml):
    """
    Calibration hand-eye avec la méthode de Tsai.
    
    Args:
        R_gripper2base: Liste des matrices de rotation flange->base (3x3)
        t_gripper2base: Liste des vecteurs de translation flange->base (3x1)
        R_target2cam: Liste des matrices de rotation target->caméra (3x3)
        t_target2cam: Liste des vecteurs de translation target->caméra (3x1)
        output_yaml: Chemin où sauvegarder la transformation
    
    Returns:
        T: Matrice de transformation homogène caméra->flange (4x4)
    """
    # Convertir les listes en tableaux NumPy
    R_grip_list = [np.array(R, dtype=np.float64) for R in R_gripper2base]
    t_grip_list = [np.array(t, dtype=np.float64).reshape(3, 1) for t in t_gripper2base]
    R_cam_list = [np.array(R, dtype=np.float64) for R in R_target2cam]
    t_cam_list = [np.array(t, dtype=np.float64).reshape(3, 1) for t in t_target2cam]
    

    # Tsai method
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_grip_list, t_grip_list,
        R_cam_list, t_cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI)
    
    T = np.eye(4)
    T[:3,:3] = R_cam2gripper
    T[:3, 3] = t_cam2gripper.flatten()

    print(T[:3, 3])
    
    with open(output_yaml, 'w') as f:
        yaml.dump({'T_cam_to_flange': T.tolist()}, f)
    print(f"Hand-eye transform saved to {output_yaml}")
    
    return T


# Charuco detection
def load_charuco(board_squares_x, board_squares_y, square_length, marker_length, aruco_dict):
              
    # aruco_dict_name est une chaîne comme "DICT_4X4_50"
    # 1) Récupère l'attribut cv2.aruco.DICT_4X4_50
    dict_attr = getattr(cv2.aruco, aruco_dict)
    # 2) Crée l'objet Dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
    # 3) Construit le CharucoBoard avec la factory
    board = cv2.aruco.CharucoBoard(
        size = (board_squares_x, board_squares_y),
        squareLength=square_length, markerLength=marker_length,
        dictionary=aruco_dict
    )
    return board

class ArucoDetector:
    def __init__(self, config_file):
        cfg = yaml.safe_load(open(config_file))
        self.camera_matrix = np.array(cfg['camera_matrix'], dtype=np.float32)
        self.dist_coeffs   = np.array(cfg['dist_coeffs'], dtype=np.float32)
        p = cfg['charuco_params']
        self.board      = load_charuco(
            p['board_squares_x'], p['board_squares_y'],
            p['square_length'], p['marker_length'],
            p['aruco_dict']           
        )
        dict_attr = getattr(cv2.aruco, p['aruco_dict'])
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
        self.detector      = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
        self.corners = None
        self.ids = None
        self.charuco_corners = None
        self.charuco_ids = None

    def detect(self, img):
        """
        Détecte le CharucoBoard et renvoie (ok, rvec, tvec).
        Utilise estimatePoseCharucoBoard plutôt qu'estimatePoseBoard.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.corners, self.ids, _ = self.detector.detectMarkers(gray)
        
        # Vérifiez d'abord si des marqueurs ont été détectés
        if self.ids is None or len(self.ids) == 0:
            self.charuco_corners = None
            self.charuco_ids = None
            return False, None, None
            
        # Raffiner la détection Charuco
        retval, self.charuco_corners, self.charuco_ids = cv2.aruco.interpolateCornersCharuco(
            self.corners, self.ids, gray, self.board)
            
        # Vérifiez si charuco_ids est None avant d'appeler len()
        if retval is not None and self.charuco_ids is not None and len(self.charuco_ids) > 3:
            # Crée les tableaux de sortie attendus par le binding Python
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
            
            # Appel avec sortie explicite
            ok = cv2.aruco.estimatePoseCharucoBoard(
                self.charuco_corners, self.charuco_ids, self.board,
                self.camera_matrix, self.dist_coeffs,
                rvec, tvec
            )
            
            if ok:
                # Vérifier que les valeurs renvoyées sont raisonnables
                if np.any(np.isnan(rvec)) or np.any(np.isnan(tvec)):
                    print("Warning: NaN values detected in pose estimation!")
                    return False, None, None
                    
                # rvec et tvec sont remplis correctement
                return True, rvec, tvec

        return False, None, None
    
    def draw_detection(self, img):
        """
        Dessine les détections sur l'image.
        Renvoie l'image avec les visuels de détection.
        """
        display_img = img.copy()
        
        # 1. Dessiner les marqueurs individuels s'ils sont détectés
        if self.corners is not None and len(self.corners) > 0:
            cv2.aruco.drawDetectedMarkers(display_img, self.corners, self.ids)
        
        # 2. Dessiner les coins Charuco s'ils sont détectés
        if self.charuco_corners is not None and len(self.charuco_corners) > 0:
            # Dessiner les coins Charuco en vert
            for corner in self.charuco_corners:
                cv2.circle(display_img, tuple(corner.astype(int).flatten()), 
                          5, (0, 255, 0), -1)
        
        return display_img



# UR5 control via RTDE
def load_hand_eye_transform(yaml_file):
    data = yaml.safe_load(open(yaml_file))
    return np.array(data['T_cam_to_flange'])

class UR5Controller:
    def __init__(self, robot_ip, hand_eye_yaml):
        import rtde_control, rtde_receive
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.T_cam2flange = load_hand_eye_transform(hand_eye_yaml)

    def move_to_pose(self, rvec, tvec, offset=(0,0,0), speed=0.1, accel=0.1):
        # Build transform marker->camera
        R_mc, _ = cv2.Rodrigues(rvec)
        T_mc = np.eye(4); T_mc[:3,:3] = R_mc; T_mc[:3,3] = tvec.flatten()
        # camera->flange
        T_fc = np.linalg.inv(self.T_cam2flange)
        # flange->base
        tcp = self.rtde_r.getActualTCPPose()
        R_fb, _ = cv2.Rodrigues(np.array(tcp[3:6]))
        T_fb = np.eye(4); T_fb[:3,:3]=R_fb; T_fb[:3,3]=tcp[:3]
        # marker->base
        T_mb = T_fb @ T_fc @ T_mc
        pos = T_mb[:3,3] + np.array(offset)
        # keep orientation
        ori = tcp[3:6]
        pose = [*pos.tolist(), *ori]
        self.rtde_c.moveL(pose, speed, accel)


# Example usage
if __name__ == '__main__':
     # ------------------------------------------------------------------
    # 1) Définition du CharucoBoard
    #    (doit correspondre EXACTEMENT à ta planche imprimée)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        size=(7,5),
        squareLength=0.04,
        markerLength=0.02,
        dictionary=aruco_dict
    )

    # ------------------------------------------------------------------
    # 2) Calibration de l'intrinsèque de la caméra
    #    Décommente pour générer `intrinsics.yaml` à partir des images
    camera_matrix, dist_coeffs = calibrate_camera_intrinsics(
        image_files='/home/robot/Bureau/Stage/electromob/testCamera/image/*.jpg',
        charuco_board=board,
        aruco_dict=aruco_dict,
        square_length=0.04,
        marker_length=0.02,
        output_yaml='intrinsics_v2.yaml')
    

    # ------------------------------------------------------------------
    # 3) Calibration hand–eye (flange→caméra)
    #    Remplis ces listes après avoir collecté ≥ 20 poses
    #R_grip2base_list = [ ... ]    # listes de matrices 3×3
    #t_grip2base_list = [ ... ]    # listes de vecteurs 3×1
    #R_target2cam_list = [ ... ]   # depuis estimatePoseCharucoBoard
    #t_target2cam_list = [ ... ]
    #T_cam2flange = calibrate_hand_eye(
    #    R_grip2base_list,
    #    t_grip2base_list,
    #    R_target2cam_list,
    #    t_target2cam_list,
    #    'hand_eye.yaml'
    #)

    # ------------------------------------------------------------------
    # 4) Initialisation des flux RealSense et des modules de détection & contrôle
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)

    # Charger les YAML générés (intrinsics + hand-eye)
    det  = ArucoDetector('intrinsics.yaml')
    ctrl = UR5Controller('10.2.30.60', 'hand_eye.yaml')

    print("Appuie sur 'Esc' pour quitter.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            ok, rvec, tvec = det.detect(img)

            # Affiche les détections et les résultats visuels
            display_img = det.draw_detection(img)
            
            # Si le Charuco est détecté complètement, dessiner les axes
            if ok:
                # Vérifier que rvec et tvec ne sont pas None et contiennent des données valides
                if rvec is None or tvec is None:
                    print("Warning: Detection OK but rvec or tvec is None!")
                    continue
                    
                # Imprimer les valeurs pour débogage
                print(f"rvec: {rvec.flatten()}")
                print(f"tvec: {tvec.flatten()}")
                
                # Dessiner les axes avec une couleur différente pour chaque axe
                cv2.drawFrameAxes(
                    display_img,
                    det.camera_matrix,
                    det.dist_coeffs,
                    rvec, tvec,
                    0.05  # longueur des axes en mètres
                )
                
                # Ajouter la position 3D au texte affiché
                position_text = f"X:{tvec[0][0]:.3f} Y:{tvec[1][0]:.3f} Z:{tvec[2][0]:.3f}"
                cv2.putText(display_img, position_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.putText(display_img, "DETECTED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Indiquer que la détection n'est pas complète
                detected_markers = len(det.corners) if det.corners is not None else 0
                cv2.putText(display_img, f"Markers: {detected_markers}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Aruco Detection', display_img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()