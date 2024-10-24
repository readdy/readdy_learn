import numpy as np


class ConversionReaction(object):
    def __init__(self, type1, type2, n_species):
        self.type1 = type1
        self.type2 = type2
        self.n_species = n_species

    def __call__(self, concentration):
        if len(concentration.shape) == 1:
            concentration = np.expand_dims(concentration, axis=0)
            result = np.zeros((1, self.n_species))
        else:
            result = np.zeros((concentration.shape[0], self.n_species))
        result[:, self.type1] += -concentration[:, self.type1]
        result[:, self.type2] += concentration[:, self.type1]
        return result.squeeze()


class FusionReaction(object):
    def __init__(self, type_from1, type_from2, type_to, n_species):
        self.type_from1 = type_from1
        self.type_from2 = type_from2
        self.type_to = type_to
        self.n_species = n_species

    def __call__(self, concentration):
        if len(concentration.shape) == 1:
            concentration = np.expand_dims(concentration, axis=0)
            result = np.zeros((1, self.n_species))
        else:
            result = np.zeros((concentration.shape[0], self.n_species))
        delta = concentration[:, self.type_from1] * concentration[:, self.type_from2]
        result[:, self.type_from1] += -delta
        result[:, self.type_from2] += -delta
        if self.type_to is not None:
            result[:, self.type_to] += delta
        return result.squeeze()


class DoubleConversion(object):
    def __init__(self, species_from, species_to, n_species):
        assert isinstance(species_from, (list, tuple)) and len(species_from) == 2
        assert isinstance(species_to, (list, tuple)) and len(species_to) == 2
        self.species_from = species_from
        self.species_to = species_to
        self.n_species = n_species

    def __call__(self, concentration):
        if len(concentration.shape) == 1:
            concentration = np.expand_dims(concentration, axis=0)
            result = np.zeros((1, self.n_species))
        else:
            result = np.zeros((concentration.shape[0], self.n_species))
        delta = concentration[:, self.species_from[0]] * concentration[:, self.species_from[1]]
        result[:, self.species_from[0]] += -delta
        result[:, self.species_from[1]] += -delta
        result[:, self.species_to[0]] += delta
        result[:, self.species_to[1]] += delta
        return result.squeeze()


class Intercept(object):
    def __init__(self, type, n_species):
        self.n_species = n_species
        self.type = type

    def __call__(self, concentration):
        if len(concentration.shape) == 1:
            result = np.zeros((1, self.n_species))
        else:
            result = np.zeros((concentration.shape[0], self.n_species))
        result[:, self.type] = 0.
        return result.squeeze()


class FissionReaction(object):
    def __init__(self, type_from, type_to1, type_to2, n_species):
        self.type_from = type_from
        self.type_to1 = type_to1
        self.type_to2 = type_to2
        self.n_species = n_species

    def __call__(self, concentration):
        if len(concentration.shape) == 1:
            concentration = np.expand_dims(concentration, axis=0)
            result = np.zeros((1, self.n_species))
        else:
            result = np.zeros((concentration.shape[0], self.n_species))
        delta = concentration[:, self.type_from]
        result[:, self.type_from] += -delta
        result[:, self.type_to1] += delta
        result[:, self.type_to2] += delta
        return result.squeeze()


class DecayReaction(object):
    def __init__(self, type_from, n_species):
        self.type_from = type_from
        self.n_species = n_species

    def __call__(self, concentration):
        if len(concentration.shape) == 1:
            concentration = np.expand_dims(concentration, axis=0)
            result = np.zeros((1, self.n_species))
        else:
            result = np.zeros((concentration.shape[0], self.n_species))
        result[:, self.type_from] = -concentration[:, self.type_from]
        return result.squeeze()


class BasisFunctionConfiguration(object):
    def __init__(self, n_species):
        self._basis_functions = []
        self._n_species = n_species

    @property
    def functions(self):
        return self._basis_functions

    @functions.setter
    def functions(self, value):
        self._basis_functions = value

    @property
    def n_basis_functions(self):
        return len(self._basis_functions)

    def add_conversion(self, type1, type2):
        self._basis_functions.append(ConversionReaction(type1, type2, self._n_species))
        return self

    def add_fusion(self, type_from1, type_from2, type_to):
        self._basis_functions.append(FusionReaction(type_from1, type_from2, type_to, self._n_species))
        return self

    def add_fission(self, type_from, type_to1, type_to2):
        self._basis_functions.append(FissionReaction(type_from, type_to1, type_to2, self._n_species))
        return self

    def add_double_conversion(self, types_from, types_to):
        self._basis_functions.append(DoubleConversion(types_from, types_to, self._n_species))

    def add_intercept(self, type):
        self._basis_functions.append(Intercept(type, self._n_species))
        return self

    def add_decay(self, type_from):
        self._basis_functions.append(DecayReaction(type_from, self._n_species))
        return self
