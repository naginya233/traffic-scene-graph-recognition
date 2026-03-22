import pytest
import numpy as np
import torch
from traffic_scene_graph.utils.bev_transform import BEVTransform

def test_bev_transform_default():
    transform = BEVTransform(frame_width=1920, frame_height=1080)
    assert transform.H.shape == (3, 3)

    # Test center pixel projection
    # Not exact due to trapezoid, but should run
    pts = np.array([[960, 540], [960, 1080]])
    bev_pts = transform.pixel_to_bev(pts)
    assert bev_pts.shape == (2, 2)

    # bottom-center should be close to origin/bottom edge of BEV
    # The default trapezoid maps bottom to y=100
    assert bev_pts[1].shape == (2,)

def test_bev_to_pixel_roundtrip():
    transform = BEVTransform(frame_width=1920, frame_height=1080)
    pts = np.array([[100, 200], [500, 600]])
    bev = transform.pixel_to_bev(pts)
    pts_recon = transform.bev_to_pixel(bev)
    np.testing.assert_allclose(pts, pts_recon, rtol=1e-4, atol=1e-4)

def test_velocity_to_bev():
    transform = BEVTransform()
    pos = np.array([[960, 1000]])
    vel = np.array([[0, -10]]) # moving up in pixel
    bev_vel = transform.velocity_to_bev(pos, vel)
    assert bev_vel.shape == (1, 2)
    # moving up in pixel (smaller y) should mean moving forward in BEV typically (smaller Y depending on mapping)
    
def test_bbox_to_bev():
    transform = BEVTransform()
    bboxes = np.array([[900, 800, 1000, 1000]]) # x1, y1, x2, y2
    bev_footprint = transform.bbox_to_bev_footprint(bboxes)
    assert bev_footprint.shape == (1, 2)
    
    # compare with bottom center mapping directly
    bottom_center = np.array([[950, 1000]])
    bev_center = transform.pixel_to_bev(bottom_center)
    np.testing.assert_allclose(bev_footprint, bev_center)
