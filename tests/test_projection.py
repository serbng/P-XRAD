import numpy as np

from pxrad.geometry.pose import DetectorPose
from pxrad.geometry.frames import Geometry, GeometryMode
from pxrad.detectors.presets import get_detector
from pxrad.geometry.projection import (
    intersection_ray_detector,
    ray_to_pixel,
)

def make_pose(geometry: Geometry, detector, det_dist=80e-3, spin=0.0) -> DetectorPose:
    return DetectorPose(
        det_dir=geometry.det_dir,
        det_norm=-geometry.det_dir,      # tua convenzione
        distance=det_dist,
        spin=spin,
        poni=detector.size / 2,          # PONI al centro in metri
    )

def test_poni_projects_to_center_for_all_modes():
    detector = get_detector("EIGER_4M")
    det_dist = 80e-3
    spin = 0.0

    rad30 = np.deg2rad(30.0)
    cos30, sin30 = np.cos(rad30), np.sin(rad30)

    for mode in GeometryMode:
        if mode is GeometryMode.CUSTOM:
            geom = Geometry(mode=mode, det_dir=np.array([sin30, 0.0, cos30]))
        else:
            geom = Geometry(mode=mode)

        pose = make_pose(geom, detector, det_dist=det_dist, spin=spin)

        # (1) Intersezione col piano deve ridare PONI in lab
        P = intersection_ray_detector(geom.det_dir, pose.poni_lab_frame(), pose.det_norm)
        assert np.allclose(P, pose.poni_lab_frame(), atol=1e-12)

        # (2) Proiezione in pixel deve dare centro immagine (shape/2)
        uv = ray_to_pixel(geom.det_dir, pose, detector)
        expected = np.array(detector.shape, dtype=float) / 2.0  # (ny,nx)/2
        assert np.allclose(uv, expected, atol=1e-10)

def test_parallel_ray_returns_nan():
    detector = get_detector("EIGER_4M")
    geom = Geometry(mode=GeometryMode.TOP_REFLECTION)
    pose = make_pose(geom, detector)

    # costruisco un raggio quasi parallelo al piano:
    # se det_norm = -det_dir, un raggio ortogonale a det_norm => d·n ~ 0
    n = pose.det_norm
    # prendo un vettore non parallelo a n e faccio cross per ottenere qualcosa nel piano
    tmp = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.99:
        tmp = np.array([1.0, 0.0, 0.0])
    d = np.cross(n, tmp)  # ~nel piano => denom ~0
    d = d / np.linalg.norm(d)

    uv = ray_to_pixel(d, pose, detector)
    assert np.all(np.isnan(uv))

def test_out_of_bounds_returns_nan():
    detector = get_detector("EIGER_4M")
    geom = Geometry(mode=GeometryMode.TOP_REFLECTION)
    pose = make_pose(geom, detector, det_dist=80e-3, spin=0.0)

    # prendo un raggio che interseca il piano, ma molto lontano dal PONI nel frame detector:
    # per semplicità, spingo lungo ex nel piano.
    # costruisco ex dal pose basis (replicando logica senza importare det_basis se vuoi)
    # qui sfruttiamo il fatto che un punto sul piano P = p00 + x*ex + y*ey
    # e costruiamo un raggio che va verso quel punto: d = unit(P)

    from pxrad.geometry.projection import det_basis_from_norm_spin
    ex, ey, _ = det_basis_from_norm_spin(pose.det_norm, pose.spin)
    p_poni = pose.poni_lab_frame()
    p00 = p_poni - pose.poni[0]*ex - pose.poni[1]*ey

    # punto molto fuori (in metri), abbastanza da uscire dal rettangolo
    big = 10.0 * detector.size[1]  # 10x larghezza
    P = p00 + big * ex

    d = P / np.linalg.norm(P)
    uv = ray_to_pixel(d, pose, detector)

    assert np.all(np.isnan(uv))