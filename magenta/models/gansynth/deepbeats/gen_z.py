import numpy as np
import os

class Gen_Z:

  mode = ''

  def generate_z_new(self, n, m):
    name = 'generate_z_' + self.mode
    method = getattr(self, name)
    return method(n, m)


  def generate_z(self, n, m):
    return np.random.normal(size=[n, m])


  def generate_z_first_units(self, n, m):
    print("generate_z:", n, m)
    res = np.zeros(shape=[n, m])

    for i in range(min(n // 4, m)):
      res[4 * i, i] = 1
      res[4 * i + 1, i] = 2
      res[4 * i + 2, i] = 3
      res[4 * i + 3, i] = -1

    return res


  def generate_z_random_units_1(self, n, m):
    print("generate_z:", n, m)
    res = np.zeros(shape=[n, m])

    for i in range(n):
      idx = np.random.randint(0, m, size=n)
      res[i, idx[i]] = 1

    return res


  def generate_z_random_units_2(self, n, m):
    print("generate_z:", n, m)
    res = np.zeros(shape=[n, m])

    for i in range(n):
      idx = np.random.randint(0, m, size=n)
      sign = np.random.randint(0, 2) * 2 - 1
      res[i, idx[i]] = sign

    return res


  def generate_z_random_units_3(self, n, m):
    print("generate_z:", n, m)
    res = np.random.normal(size=[n, m])

    for i in range(n):
      vec = res[i, :]
      norm = np.linalg.norm(vec)
      res[i, :] = vec / norm

    return res


  def generate_z_arithmetic_radius(self, n, m, direc=None):
    print("generate_z_arithmetic_radius:", n, m)
    res = np.zeros(shape=[n, m])
    if direc is None:
      direc = np.random.normal(size=[1, m])

    direc /= np.linalg.norm(direc)

    for i in range(n):
      res[i, :] = i * direc

    return res


  def generate_z_square_radius(self, n, m, direc=None):
    print("generate_z_square_radius:", n, m)
    res = np.zeros(shape=[n, m])
    if direc is None:
      direc = np.random.normal(size=[1, m])

    direc /= np.linalg.norm(direc)

    for i in range(n):
      res[i, :] = i * i * direc

    return res


  def generate_z_exp_radius(self, n, m, direc=None):
    print("generate_z_exp_radius:", n, m)
    res = np.zeros(shape=[n, m])
    if direc is None:
      direc = np.random.normal(size=[1, m])

    direc /= np.linalg.norm(direc)

    for i in range(1, n):
      res[i, :] = 2 ** (i - 1) * direc

    return res


  def generate_z_arithmetic_diameter(self, n, m, direc=None):
    print("generate_z_arithmetic_diameter:", n, m)
    res = np.zeros(shape=[n, m])

    if direc is None:
      direc = np.random.normal(size=[1, m])

    direc /= np.linalg.norm(direc)

    for i in range(n):
      res[i, :] = (i - n / 2) * direc

    return res


  def generate_z_grow_ball(self, n, m, c=None):
    if c is None:
      c = np.zeros(shape=[1, m])

    print("generate_z_grow_ball:", n, m)
    res = np.zeros(shape=[n, m])

    for i in range(n):
      direc = np.random.normal(size=[1, m])
      direc /= np.linalg.norm(direc)
      res[i, :] = c + i / n * direc

    return res


  def write_matrix_to_file(self, mat, folder_path, name_prefix):
    name = name_prefix + "_coord_z.txt"
    with open(os.path.join(folder_path, name), 'wb') as f:
      np.savetxt(f, mat, fmt='%.2f')

# NTM
