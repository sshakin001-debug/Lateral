"""Mock ROS modules for testing without ROS installation"""
import sys


class MockPointCloud2:
    """Mock PointCloud2 message"""
    pass


class MockMsg:
    """Mock message class"""
    PointCloud2 = MockPointCloud2


class MockSensorMsgs:
    """Mock sensor_msgs module"""
    msg = MockMsg()


# Mock the entire sensor_msgs module
sys.modules['sensor_msgs'] = MockSensorMsgs()
sys.modules['sensor_msgs.msg'] = MockMsg()

# Also mock any other ROS modules that might be imported
sys.modules['geometry_msgs'] = type(sys)('geometry_msgs')
sys.modules['std_msgs'] = type(sys)('std_msgs')
