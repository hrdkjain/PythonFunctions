
import numpy as np
import os
import natsort
import scipy.spatial
from plyfile import PlyData

def writeOff(file, im):
  if not file.endswith('.off'):
      file += '.off'

  if len(im.shape)>2:
    if len(im.shape)==4:
      im = np.squeeze(im,axis=0)
    f = open(file,'w+')
    f.write('OFF\n%d 0 0\n' %(np.size(im,0)*np.size(im,1)))
    for r in range(im.shape[1]):
        for c in range(im.shape[0]):
            f.write('%.5f %.5f %.5f\n' %(im[c][r][2],im[c][r][1],im[c][r][0]))
    f.close()
  elif len(im.shape)==2:
    f = open(file,'w+')
    if im.shape[0] == 3:
      im = np.transpose(im)
    f.write('OFF\n%d 0 0\n' %(np.size(im,0)))
    for c in range(im.shape[0]):
      f.write('%.5f %.5f %.5f\n' %(im[c][0],im[c][1],im[c][2]))
    f.close()

def writeOffMesh(file, vertices, faces):
  if vertices.shape[1] != 3:
    vertices = np.transpose(vertices)

  if faces.shape[1] != 3:
    faces = np.transpose(faces)
  if faces.min() == 1:
    # subrtract 1 to make zero index
    faces -= 1

  if not file.endswith('.off'):
      file += '.off'
  f = open(file,'w+')
  f.write('OFF\n%d %d 0\n' % (len(vertices), len(faces)))
  for id in range(len(vertices)):
    f.write('%.5f %.5f %.5f\n' % (vertices[id, 0], vertices[id, 1], vertices[id, 2]))
  for id in range(len(faces)):
    f.write('3 %d %d %d\n' % (faces[id, 0], faces[id, 1], faces[id, 2]))
  f.close()
    
def bounding_box(points):
  x = [p[0] for p in points]
  y = [p[1] for p in points]
  z = [p[2] for p in points]
  return (min(x),max(x)),(min(y),max(y)),(min(z),max(z)) 
  
def bounding_box_filter(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(bound_x, bound_y, bound_z)

    return bb_filter

def locate_center(points):
  x = [p[0] for p in points]
  y = [p[1] for p in points]
  z = [p[2] for p in points]
  centroid = (sum(x) / len(points), sum(y) / len(points), sum(z) / len(points))
  return centroid

def translate(points, translation):
  output = points
  for i in range(len(points)):
    for j in range(len(points[i,:])):
      output[i,j] = points[i,j]+translation[j]
  return output#[x,y,z]

def read_off(fname):
    with open(fname, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, n_edges = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)

def read_ply(fname):
    plydata = PlyData.read(fname)
    return np.asarray(np.transpose(np.vstack((plydata['vertex'].data['x'],plydata['vertex'].data['y'],plydata['vertex'].data['z']))), dtype=np.float32), \
           np.asarray(np.vstack(plydata['face'].data['vertex_indices']), dtype=np.int64)

def in_docker():
    """ Returns: True if running in a Docker container, else False """
    with open('/proc/1/cgroup', 'rt') as ifh:
        return 'docker' in ifh.read()

def get_new_name(filePath, ext):
    count = 0
    outFile = filePath + ext
    while os.path.exists(outFile):
        count = count+1
        outFile = os.path.join(filePath + '_' + str(count) + ext)
    return outFile

def extractNumber(string):
    # Extract number from a string with characters and numbers
    try:
        return int(string)
    except:
        numberStartId = None
        for i in range(len(string)):
            try:
                int(string[i])
                numberStartId = i
                break
            except:
                pass
        return int(string[numberStartId:])

def euclideanDistance(a,b):
    # Compute euclidean distance for 3d arrays
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[0] == 3:
        a = np.transpose(a)

    if b.shape[0] == 3:
        b = np.transpose(b)
    assert a.shape == b.shape

    # Now the inputs have shape nx3
    dist = np.empty((a.shape[0]))
    for i in range(a.shape[0]):
        dist[i] = scipy.spatial.distance.euclidean(a[i],b[i])

    return dist

def listAllFiles(dir, ext):
    fileList = []
    for root, dirs, files in os.walk(dir):
        for filename in files:
            if filename.endswith(tuple(ext)):
                fileList.append(os.path.join(root, filename))
    fileList = natsort.natsorted(fileList)
    return fileList

def read_mesh(fname):
    if fname.endswith('.off'):
        return read_off(fname)
    elif fname.endswith('.ply'):
        return read_ply(fname)
    else:
        raise ValueError('file extension not handled')
