import numpy as np

def camerasCommunication(connection, n_cameras):
    global ADJ
    AJD = np.ones((n_cameras, n_cameras))       
    
def checkNeighbour(groupCameras, node1, node2):
    """
    Check if both nodes are neigbours
    """
    neighbours = True
    out = ADJ[node1][node2]
    
    if out == 0:
        neighbours = False
    return neighbours

def CrossCamerasIdentities(distributed_gallery, current_camera, id_cameras):
    
    
    