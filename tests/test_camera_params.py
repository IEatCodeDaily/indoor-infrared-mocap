import pytest
import json
import numpy as np
from pathlib import Path
from irtracking.params import CameraParamsManager, IntrinsicParams, ExtrinsicParams
from typing import Dict
import os


@pytest.fixture(scope='session')
def sample_intrinsic_data():
    return [
        {
            "camera_id": 0,
            "intrinsic_matrix": [[539.3649, 0, 344.2623],
                                [0, 539.5124, 242.5057],
                                [0, 0, 1]],
            "distortion_coef": [-0.1046, 0.1401, 0, 0, 0],
            "rotation": 0
        },
        {
            "camera_id": 1,
            "intrinsic_matrix": [[542.2715, 0, 328.9326],
                                [0, 542.4454, 244.2944],
                                [0, 0, 1]],
            "distortion_coef": [-0.1170, 0.1578, 0, 0, 0],
            "rotation": 0
        }
    ]

@pytest.fixture(scope='session')
def sample_extrinsic_data():
    return [{
        "camera_id": 0,
        "rotation": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ],
        "position": [0, 0, 0]
    },
    {
        "camera_id": 1, 
        "rotation": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ],
        "position": [0, 0, 0]
    }
    ]  


@pytest.fixture
def sample_intrinsic_params(sample_intrinsic_data):
    return dict((data['camera_id'], IntrinsicParams.from_dict(data)) for data in sample_intrinsic_data)

@pytest.fixture
def sample_extrinsic_params(sample_extrinsic_data):
    return dict((data['camera_id'], ExtrinsicParams.from_dict(data)) for data in sample_extrinsic_data)

@pytest.fixture(scope='session')
def temp_config_dir(tmp_path_factory, sample_intrinsic_data, sample_extrinsic_data):
    config_path = tmp_path_factory.mktemp("config")
    intrinsic_file = config_path / "camera-intrinsic.json"
    with intrinsic_file.open('w') as f:
        json.dump(sample_intrinsic_data, f)
    extrinsic_file = config_path / "camera-extrinsic.json"
    with extrinsic_file.open('w') as f:
        json.dump(sample_extrinsic_data, f)
    return config_path

@pytest.fixture
def params_manager(temp_config_dir):
    return CameraParamsManager(config_dir=temp_config_dir)

class TestCameraParamsManager:
    def test_init(self, params_manager, temp_config_dir, sample_intrinsic_params, sample_extrinsic_params):
        assert params_manager.config_dir == temp_config_dir
        assert params_manager.intrinsic.keys() == sample_intrinsic_params.keys()
        assert params_manager.extrinsic.keys() == sample_extrinsic_params.keys()
        
        for key in sample_intrinsic_params:
            assert np.array_equal(params_manager.intrinsic[key].matrix, sample_intrinsic_params[key].matrix)
            assert np.array_equal(params_manager.intrinsic[key].distortion, sample_intrinsic_params[key].distortion)
        
        for key in sample_extrinsic_params:
            assert np.array_equal(params_manager.extrinsic[key].position, sample_extrinsic_params[key].position)
            assert np.array_equal(params_manager.extrinsic[key].rotation, sample_extrinsic_params[key].rotation)
        
    def test_load_intrinsic_params(self, params_manager, sample_intrinsic_params):            
        # Test loading
        params_manager.load_intrinsic_params()
        assert params_manager.intrinsic.keys() == sample_intrinsic_params.keys()
        
        for key in sample_intrinsic_params:
            assert np.array_equal(params_manager.intrinsic[key].matrix, sample_intrinsic_params[key].matrix)
            assert np.array_equal(params_manager.intrinsic[key].distortion, sample_intrinsic_params[key].distortion)
        
    # def test_save_intrinsic_params(self, params_manager, sample_intrinsic_data):
    #     # Set data
    #     params_manager.intrinsic.matrix = np.array(sample_intrinsic_data[0]['intrinsic_matrix'])
    #     params_manager.intrinsic.distortion = np.array(sample_intrinsic_data[0]['distortion_coef'])
        
    #     # Save and reload
    #     params_manager.save_intrinsic()
    #     loaded_params = params_manager.load_intrinsic()
        
    #     # Verify
    #     np.testing.assert_array_almost_equal(
    #         loaded_params[0]['intrinsic_matrix'],
    #         params_manager.intrinsic.matrix
    #     )
        
    # def test_missing_file(self, params_manager):
    #     with pytest.raises(FileNotFoundError):
    #         params_manager.load_intrinsic_params("nonexistent.json")
            
    # def test_invalid_json(self, temp_config_dir, params_manager):
    #     # Create invalid JSON file
    #     config_file = temp_config_dir / "invalid.json"
    #     with config_file.open('w') as f:
    #         f.write("invalid json")
            
    #     with pytest.raises(json.JSONDecodeError):
    #         params_manager.load_intrinsic_params("invalid.json")
            
    # def test_invalid_parameter_format(self, temp_config_dir, params_manager):
    #     # Create file with wrong format
    #     config_file = temp_config_dir / "wrong-format.json"
    #     with config_file.open('w') as f:
    #         json.dump([{"wrong_key": [1, 2, 3]}], f)
            
    #     with pytest.raises(KeyError):
    #         params_manager.load_intrinsic_params("wrong-format.json")