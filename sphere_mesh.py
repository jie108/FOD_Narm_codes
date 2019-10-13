import numpy as np
from scipy.sparse import coo_matrix

# generate a simple triangulation (icosphere)
# migrated from matlab package "toolbox_wavelet_meshes" by Gabriel Peyre
def compute_base_mesh():

	tau = 0.8506508084
	one = 0.5257311121
	vertex = np.array(
		[[tau, one, 0], [-tau, one, 0], [-tau, -one, 0], [tau, -one, 0], 
		[one, 0, tau], [one, 0, -tau], [-one, 0, -tau], [-one, 0, tau],
		[0, tau, one], [0, -tau, one], [0, -tau, -one], [0, tau, -one]]).T
	face = np.array(
		[[4, 8, 7], [4, 7, 9], [5, 6, 11], [5, 10, 6], [0, 4, 3], 
		[0, 3, 5], [2, 7, 1], [2, 1, 6], [8, 0, 11], [8, 11, 1], 
		[9, 10, 3], [9, 2, 10], [8, 4, 0], [11, 0, 5], [4, 9, 3], 
		[5, 3, 10], [7, 8, 1], [6, 1, 11], [7, 2, 9], [6, 10, 2]]).T

	return vertex, face

# perform a mesh sub-division
# migrated from matlab package "toolbox_wavelet_meshes" by Gabriel Peyre
def perform_mesh_subdivision(vertex, face):

	n = vertex.shape[1]

	# 1:4 tolopoligical sub-division with linear interpolation
	i = np.concatenate((face[0], face[1], face[2], face[1], face[2], face[0]))
	j = np.concatenate((face[1], face[2], face[0], face[0], face[1], face[2]))
	I = np.where(i<j)[0]
	i, j = i[I], j[I]
	_, I = np.unique(i+1234567*j, return_index=True)
	i, j = i[I], j[I]
	s = n+np.arange(len(i))  # len(i) is number of edges

	A = coo_matrix((np.concatenate((s, s)), (np.concatenate((i, j)), np.concatenate((j, i)))), shape = (n, n)).toarray()

	# first face
	v01, v12, v20 = A[face[0], face[1]], A[face[1], face[2]], A[face[2], face[0]]
	face = np.hstack((np.vstack((face[0], v01, v20)), np.vstack((face[1], v12, v01)), np.vstack((face[2], v20, v12)), 
		np.vstack((v01, v12, v20))))

	# add new vertices at the edges center
	vertex = np.hstack((vertex, (vertex[:, i] + vertex[:, j])/2))

	# project on the sphere
	vertex /= np.sqrt(np.sum(vertex**2, 0))

	return vertex, face

# smooth a function defined on a mesh by averaging
# migrated from matlab package "toolbox_wavelet_meshes" by Gabriel Peyre
def perform_mesh_smoothing(vertex, face):

	# compute normalized averaging matrix (add diagonal)
	# first compute the adjacency matrix of a given triangulation
	W = coo_matrix((np.ones(face.shape[1]*6), (np.concatenate((face[0], face[1], face[2], face[1], face[2], face[0])), 
		np.concatenate((face[1], face[2], face[0], face[0], face[1], face[2]))))).toarray()
	W = (W > 0).astype("double") + np.identity(np.max(face)+1)
	D = np.diag(1/np.sum(W, 0))
	W = D.dot(W)

	# do averaging to smooth the field
	vertex = vertex.dot(W.T)
	vertex /= np.sqrt(np.sum(vertex**2, 0))

	return vertex

# compute a semi-regular sphere (J is the level of sub-division) (set relaxation>0 to enhance the positions)
# migrated from matlab package "toolbox_wavelet_meshes" by Gabriel Peyre
def compute_semiregular_sphere(J, relaxation):

	vertex, face = compute_base_mesh()
	for j in range(J-1):
		vertex, face = perform_mesh_subdivision(vertex, face)
		# do some smoothing
		if relaxation > 0:
			for i in range(relaxation):
				vertex = perform_mesh_smoothing(vertex, face)

	return vertex, face

# generate equal-angle grid with level J on entire or half sphere (set relaxation>0 to enhance the positions)
# pos: 3D coordinates of equal-angle grid
# theta, phi: spherical coordinates of equal-angle grid (theta: polar angle, phi: azimuthal angle)
# sampling_index: index of grid points on half sphere if half=True
def spmesh(J, half, relaxation = 1):

	if J == 2.5:
		pos, _ = compute_semiregular_sphere(3, relaxation)
	else:
		pos, _ = compute_semiregular_sphere(J, relaxation)
	pos[np.abs(pos)<1e-15] = 0

	if half:
		sampling_index = []
		for i in range(pos.shape[1]):
			if pos[2, i] == 0:
				if pos[1, i] == 0:
					if pos[0, i] > 0:
						sampling_index.append(i)
				elif pos[1, i] > 0:
					sampling_index.append(i)
			elif pos[2, i] > 0:
				sampling_index.append(i)
		pos = pos[:, sampling_index]

	# spherical coordinates of vertex
	# theta: polar angle, phi: azimuthal angle
	theta = np.arccos(pos[2])
	phi = np.arctan2(pos[1], pos[0])
	phi += 2*np.pi*(phi<0)

	# take 41 out of 81 directions: at each level of theta, take around half phi
	if J == 2.5:
		phi[37] = 2*np.pi  # make consistent with the "bug" in matlab code
		sampling_index = []
		end_point = np.array([0, 10, 20, 40, 50, 70, 85, 91])/180.0*np.pi
		for i in range(len(end_point)-1):
			index_tmp = np.where((theta>=end_point[i]) & (theta<end_point[i+1]))[0]
			sampling_index += index_tmp[np.argsort(phi[index_tmp])][np.arange(0, len(index_tmp), 2)].tolist()
		sampling_index.sort()
		pos = pos[:, sampling_index]
		theta = theta[sampling_index]
		phi = phi[sampling_index]

	if half:
		return pos, theta, phi, sampling_index
	else:
		return pos, theta, phi