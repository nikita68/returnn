import subprocess
import h5py
from scipy.io.netcdf import NetCDFFile

def cmd(cmd):
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, close_fds=True)
  result = [ tag.strip() for tag in p.communicate()[0].split('\n')[:-1]]
  p.stdout.close()
  return result

def hdf5_dimension(filename, dimension):
  fin = h5py.File(filename, "r")
  res = fin.attrs[dimension]
  fin.close()
  return res

def hdf5_strings(handle, name, data):
  S=max([len(d) for d in data])
  dset = handle.create_dataset(name, (len(data),), dtype="S"+str(S))
  dset[...] = data

def strtoact(act):
  import theano.tensor as T
  activations = { 'logistic' : T.nnet.sigmoid,
                  'tanh' : T.tanh,
                  'relu': lambda z : (T.sgn(z) + 1) * z * 0.5,
                  'identity' : lambda z : z,
                  'one' : lambda z : 1,
                  'zero' : lambda z : 0,
                  'softsign': lambda z : z / (1.0 + abs(z)),
                  'softsquare': lambda z : 1 / (1.0 + z * z),
                  'maxout': lambda z : T.max(z, axis = 0),
                  'sin' : T.sin,
                  'cos' : T.cos }
  assert activations.has_key(act), "invalid activation function: " + act
  return activations[act]
