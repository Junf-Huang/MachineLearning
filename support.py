import struct
import gzip
import pickle


MAGIC = b'AIB\x00'
FORMAT_VERSION = b'\x00\x01'


def convert_celsius(fahrenheit):
    print("celsius is {}.".format((fahrenheit-32)/1.8))


def convert_fahrenheit(celsius):
    print("fahrenheit is {}.".format(celsius*1.8+32))

