import numpy as np

from pxrad.geometry.pose import DetectorPose
from pxrad.geometry.frames import GeometryMode, Geometry, LAB_FRAME
from pxrad.detectors.presets import get_detector
from pxrad.geometry.projection import (
    det_basis_from_norm_spin, 
    intersection_ray_detector, 
    ray_to_pixel
)

from textcolors import tcolor

detector = get_detector("EIGER_4M")
det_dist = 80e-3
spin = 0

# For CUSTOM GeometryMode, detector at 30°
rad30 = 30*np.pi/180
cos30, sin30 = np.cos(rad30), np.sin(rad30)

print(
    f"{tcolor.HEADER}LABORATORY FRAME REFERENCE {tcolor.ENDC} \n"
    f"x_hat: {LAB_FRAME.x_hat}\n"
    f"y_hat: {LAB_FRAME.y_hat}\n"
    f"z_hat: {LAB_FRAME.z_hat}\n"
)

print(
    f"{tcolor.HEADER}Detector used: {detector.name}{tcolor.ENDC}\n"
    f"{tcolor.WARNING}Default detector distance: {det_dist} [m]{tcolor.ENDC}\n"
    f"{tcolor.WARNING}Default detector spin: {spin} [rad]{tcolor.ENDC}\n"
)

print(f"{tcolor.BOLD}Testing default geometry modes...{tcolor.ENDC}\n")
for mode in GeometryMode:
    if mode.name == "CUSTOM":
        geometry = Geometry(mode=mode, det_dir=np.array([sin30, 0, cos30]))
    else:
        geometry = Geometry(mode=mode)
    
    print(tcolor.OKBLUE + mode.name + tcolor.ENDC)
    
    pose = DetectorPose(
        det_dir=geometry.det_dir,
        det_norm=-geometry.det_dir,
        distance=det_dist,
        spin=spin,
        poni=detector.size/2
    )
    
    print(
        "Nominal detector pose:\n"
        f"det_dir: {pose.det_dir}\n"
        f"det_norm: {pose.det_norm}\n"
        f"poni: {pose.poni} [m]\n"
        f"poni: {np.array(detector.shape)/2} [px] \n"
        f"poni in LAB FRAME {pose.poni_lab_frame()}\n"
    )
    
    ex, ey, ez = det_basis_from_norm_spin(pose.det_norm, pose.spin)
    
    print(
        "Detector basis IN LABORATORY FRAME:\n"
        f"ex: {ex}\n"
        f"ey: {ey}\n"
        f"ez: {ez}\n"
    )
    
    poni_lab_frame = intersection_ray_detector(geometry.det_dir, pose.poni_lab_frame(), pose.det_norm)
    
    poni_px = ray_to_pixel(geometry.det_dir, pose, detector)
    
    proj_status_lab = "OK" if np.allclose(poni_lab_frame, pose.poni_lab_frame()) else "NOT OK"
    proj_status_det = "OK" if np.allclose(poni_px, np.array(detector.shape)/2) else "NOT OK"
    
    print(
        "Checking projections\n"
        f"poni in LAB frame: {tcolor.FAIL}{proj_status_lab}{tcolor.ENDC}\n"
        f"    - nominal: {pose.poni_lab_frame()}\n"
        f"    - projected: {poni_lab_frame}\n"
        f"poni in px units: {tcolor.FAIL}{proj_status_det}{tcolor.ENDC}\n"
        f"    - nominal: {np.array(detector.shape)/2}\n"
        f"    - projected: {poni_px}\n"
    )