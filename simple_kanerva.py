import numpy

import struct
import sys


def floatToBits(f):
    # http://stackoverflow.com/questions/14431170/get-the-bits-of-a-float-in-python
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]


class KanervaCoding():

    def __init__(self, low_corner, high_corner, num_features, tiles=False, distance_metric='hamming', random_seed=None, distribution=numpy.random.uniform, bias=False):
        numpy.random.seed(random_seed)

        assert len(low_corner) == len(high_corner)
        self.low_corner = low_corner
        self.high_corner = high_corner

        self.num_features = num_features
        self.num_prototypes = num_features - 1 if bias else num_features

        assert self.num_prototypes > 2

        self.distance_metric = distance_metric
        self.num_dims = len(low_corner)
        self.bias = bias

        if tiles:
            columns = [numpy.linspace(low_corner[dim], high_corner[dim], self.num_prototypes) for dim in range(self.num_dims)]
        elif type(distribution) == list:
            assert len(distribution) == self.num_dims
            columns = [distribution[dim](low_corner[dim], high_corner[dim], self.num_prototypes) for dim in
                       range(self.num_dims)]
        else:
            columns = [distribution(low_corner[dim], high_corner[dim], self.num_prototypes) for dim in
                       range(self.num_dims)]
        columns = numpy.array(columns)
        self.prototypes = numpy.transpose(columns)

    def get_x(self, sensor_reading, num_on_features, ignore=None, distance_metric='hamming'):
        assert len(sensor_reading) == self.num_dims
        # ignore is either None or a list of dimension indices, 0-indexed, to not include in the distance computation
        if not ignore:
            ignore = []
        not_ignored = numpy.array(list(set(range(self.num_dims)).difference(ignore)))

        distance_metric_box = [distance_metric, self.distance_metric]

        if 'euclidean' in distance_metric_box:
            temp_prototypes = self.prototypes
            temp_sensor_reading = sensor_reading
        elif 'scaled-euclidean' in distance_metric_box:
            temp_prototypes = (self.prototypes - self.low_corner) / [(self.high_corner[dim] - self.low_corner[dim]) for dim
                in range(self.num_dims)]
            temp_sensor_reading = (numpy.array(sensor_reading) - self.low_corner) / [(self.high_corner[dim] - self.low_corner[dim]) for dim
                in range(self.num_dims)]
        if 'euclidean' in distance_metric_box or 'scaled-euclidean' in distance_metric_box:
            distances = numpy.array(
                [numpy.linalg.norm((temp_prototypes[i] - temp_sensor_reading)[not_ignored]) for i in range(self.num_prototypes)])
        else:   # default to Hamming-distance
            distances = []
            for proto in self.prototypes:
                # take the XOR and count the 1s, then sum over all num_dim dimensions.
                distances.append(sum([bin(floatToBits(sensor_reading[dim]) ^ floatToBits(proto[dim])).count('1') if dim not in ignore else 0 for dim in range(self.num_dims)] ))
            distances = numpy.array(distances)

        # Return the indices of the prototypes with the smallest distance from the sensor_reading
        num_on_prototypes = num_on_features - 1 if self.bias else num_on_features
        on_prototypes = numpy.argpartition(-distances, -num_on_prototypes)[-num_on_prototypes:]

        if self.bias:
            on_prototypes = numpy.append(on_prototypes, self.num_prototypes)
        return on_prototypes

    def get_num_features(self):
        return self.num_features

def kanerva_normal(low, high, num_prototypes):
    width = high-low
    midpoint = low+(width/2)
    sd = width/7
    return numpy.random.normal(loc=midpoint, scale=sd, size=num_prototypes)


if __name__ == '__main__':
    coding = KanervaCoding([200], [824], 20, tiles=True, distance_metric='euclidean')
    print 510, coding.get_x([510], 1)
    print 520, coding.get_x([520], 1)