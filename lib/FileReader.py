# -*- coding: utf-8 -*-
"""
Created on Wed May  3 13:59:07 2023

@author: grundch
"""
import numpy as np
import datetime
import struct
import cv2

class FileReader():


    def __init__(self):
        pass
    
    def loadFile(self, fileLoc):
        if fileLoc.endswith(".asc"):
            header, data2D = self.loadASCFile(fileLoc)
            
        if fileLoc.endswith(".bcrf"):
            header, data2D = self.loadBCRFFile(fileLoc)
            
        if fileLoc.endswith(".png"):
            header, data2D = self.loadPNGFile(fileLoc)
        
        if fileLoc.endswith(".npy"):
            header, data2D = self.loadNPYFile(fileLoc)
            
        return header, data2D
    
    # code for .npy file
    #--------------------------------------------------------------------------
    def loadNPYFile(self, fileLoc):
        header = self.readHeaderFromNPYFile(fileLoc)
        data = self.readDataFromNPYFile(fileLoc)
        return header, data

    def readHeaderFromNPYFile(self, fileLoc):
        """
        Function to extract the header of a .png file into a dictionary.

        Parameters
        ----------
        fileLoc : str
            Location of the .asc file.

        Returns
        -------
        header : dict
            Extracted header data.

        """
        return None

    def readDataFromNPYFile(self, fileLoc):
        """
        Function to extract the data from a .png file into a numpy array.

        Parameters
        ----------
        fileLoc : str
            Location of the .asc file.

        Returns
        -------
        data : array
            Extracted data.

        """
        data = np.load(fileLoc)
        return data

    # code for .png file
    #--------------------------------------------------------------------------
    def loadPNGFile(self, fileLoc):
        header = self.readHeaderFromPNGFile(fileLoc)
        data = self.readDataFromPNGFile(fileLoc)
        return header, data

    def readHeaderFromPNGFile(self, fileLoc):
        """
        Function to extract the header of a .png file into a dictionary.

        Parameters
        ----------
        fileLoc : str
            Location of the .asc file.

        Returns
        -------
        header : dict
            Extracted header data.

        """
        return None

    def readDataFromPNGFile(self, fileLoc):
        """
        Function to extract the data from a .png file into a numpy array.

        Parameters
        ----------
        fileLoc : str
            Location of the .asc file.

        Returns
        -------
        data : array
            Extracted data.

        """
        data = cv2.imread(fileLoc)
        return np.array(data)
    
    # code for .bcrf file
    #--------------------------------------------------------------------------
    def loadBCRFFile(self, path):
        header = self.readHeaderFromBCRFFile(path)
        data = self.readDataFromBCRFFile(path)
        data2D = data.reshape(int(header['xpixels']), int(header['ypixels']))
        return header, data2D

    def readHeaderFromBCRFFile(self, fileLoc):
        """
        Function to extract the header of a .bcrf file into a dictionary.

        Parameters
        ----------
        fileLoc : str
            Location of the .asc file.

        Returns
        -------
        header : dict
            Extracted header data.

        """
        with open(fileLoc, 'br') as f:
            lineCount = 0
            header = {}
            lastLineInHeader = b'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              \n'

            line = b''
            while lineCount < 13:
                line = f.readline()
                if line ==  lastLineInHeader:
                    break
                elif b'starttime' in line:
                    line = line.decode("utf-8")
                    line = line.split(' ')
                    key = line[0]
                    dt = datetime.datetime.strptime("{}/{}/{} {}".format(line[3],line[2],line[1], line[4][:-1]), "%y/%m/%d %H:%M:%S")
                    header[key] = dt
                else:
                    key, value = line.split(b' = ')
                    try:
                        header[key.decode("utf-8")] = float(value)
                    except:
                        header[key.decode("utf-8")] = value.decode("utf-8")
                lineCount += 1
        return header

    def readDataFromBCRFFile(self, fileLoc):
        """
        Function to extract the data from a .bcrf file into a numpy array.

        Parameters
        ----------
        fileLoc : str
            Location of the .asc file.

        Returns
        -------
        data : array
            Extracted data.

        """
        with open(fileLoc, 'br') as f:
            lineCount = -1
            while lineCount < 14:
                lineCount += 1
                continue
            dataLine = f.readline()
            length = 4
            byte = dataLine[:length]
            i=length
            data = []
            while byte != b"":
                # Do stuff with byte.
                data.append(struct.unpack('f', byte))
                byte = dataLine[i:i+length]
                i=i+length
        return np.array(data)

    # code for .asc file
    #--------------------------------------------------------------------------
    def loadASCFile(self, path):
        header = self.readHeaderFromASCFile(path)
        data = self.readDataFromASCFile(path)
        data2D = data.reshape(int(header['x-pixels']), int(header['y-pixels']))
        return header, data2D
    
    def readHeaderFromASCFile(fileLoc):
        """
        Function to extract the header of a .asc file into a dictionary.

        Parameters
        ----------
        fileLoc : str
            Location of the .asc file.

        Returns
        -------
        header : dict
            Extracted header data.

        """
        with open(fileLoc, 'r') as f:
            lineCount = 0
            header = {}
            line = f.readline()
            line = f.readline()
            header['Date'] = line.split('T')[0][2:]
            header['Time'] = line.split('T')[1]
            while lineCount < 11:
                line = f.readline()
                line = line.split(':')
                try:
                    header[line[0][2:]] = float(line[1])
                except:
                    if 'unit' in line[0][2:]:
                        header[line[0][2:]] = str(line[1][4:-1])
                    else:
                        header[line[0][2:]] = str(line[1][3:-1])
                lineCount += 1
            return header

    def readDataFromASCFile(fileLoc):
        """
        Function to extract the data from a .asc file into a numpy array.

        Parameters
        ----------
        fileLoc : str
            Location of the .asc file.

        Returns
        -------
        data : array
            Extracted data.

        """
        with open(fileLoc, 'r') as f:
            lineCount = -1
            data = []
            for line in f:
                lineCount += 1
                if lineCount < 14:
                    continue
                data.append(float(line[:-1]))
        return np.array(data)