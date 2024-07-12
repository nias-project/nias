from nias.bindings.nias_cpp.interfaces import Vector


class NiasCppVector(Vector):
    def __init__(self, impl):
        self.impl = impl
        self.dim = len(impl)

    def __eq__(self, other):
        return type(self) == type(other) and self.impl == other.impl

    def __getitem__(self, ind):
        return self.impl[ind]

    def __setitem__(self, ind, val):
        self.impl[ind] = val

    def __len__(self):
        return self.dim

    def __iter__(self):
        for i in range(self.dim):
            yield self[i]

    def copy(self, deep=False):
        return NiasCppVector(self.impl.copy())

    def scal(self, alpha):
        self.impl.scal(alpha)

    def axpy(self, alpha, x):
        self.impl.axpy(alpha, x.impl)

    def dot(self, other):
        return self.impl.dot(other.impl)
