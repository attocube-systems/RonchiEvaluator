from cv2 import threshold
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import cv2


class Evaluator():


    def __init__(self, resolution, numPeaks, height, FWHMRangeAccepted = [200, 700]):
        
        self.resolution = resolution
        self.numPeaks = numPeaks
        self.height = height
        self.FWHMRangeAccepted = FWHMRangeAccepted

    def addLinesToImage(self, file):
        fileCopy = file
        m = self.get_direction(file)
        P_start, P_end = self._getPoints_new(file, m, skip = 5)
        lines = self._getLines(P_start, P_end, skip=0) #TODO: Test with and without skip, maybe later implement, that user can choose skip in GUI
        dataWithLines = self._drawLinesToData(lines, fileCopy)
        del fileCopy
        return dataWithLines

    def evaluate(self, data):
        if len(np.shape(data)) == 2:
            sizeX,sizeY = np.shape(data)
        elif len(np.shape(data)) == 3:
            sizeX,sizeY,z = np.shape(data)
        else:
            print('ERROR')
        m = self.get_direction(data)
        P_start, P_end = self._getPoints_new(data, m, skip = 5)
        lines = self._getLines(P_start, P_end, skip=0)

         #TODO: arbitrary values, check if this can be automised
        snip = 20
        
        HalfMaxes = []
        meanPeaks = []
        allX = []
        gausses = []
        dgdxes = []
        lineCounter = 0
        for line in lines:
            lineCounter  += 1
            check = False
            for p in line:
                if p[0] > sizeX:
                    check = True
                    break
                if p[1] > sizeY:
                    check = True
                    break
            if check:
                continue
                
            lineData = self.getDataOnLine(data, line)
            x = np.arange(0, len(lineData))*self.resolution
            dx = x[1]-x[0]

            grad_lineData = np.gradient(lineData, dx)
            grad_lineData = grad_lineData/max(grad_lineData)

            peaks = self.find_line_peaks(grad_lineData, self.numPeaks, self.height)
            
            
            for p in peaks:
                if (p[1]<snip) or (p[1]+snip>x.shape[0]):
                    continue
                x_g = x[p[1]-snip:p[1]+snip]
                dgdx = grad_lineData[p[1]-snip:p[1]+snip]

                mean_g = sum(x_g * dgdx) / sum(dgdx)
                sigma_g = np.sqrt(sum(dgdx * (x_g - mean_g)**2) / sum(dgdx))
                try:
                    popt,pcov = curve_fit(self.Gauss, x_g, dgdx, p0=[max(dgdx), mean_g, sigma_g], maxfev = 5000)
                except RuntimeError as e:
                    continue
                fw = self.FWHM(*popt)

                if np.isnan(fw):
                    print(str(lineCounter) + ' fw: ' + str(fw))
                    continue
                elif fw > self.FWHMRangeAccepted[1]:
                    print(str(lineCounter) + ' fw: ' + str(fw))
                    continue
                elif fw < self.FWHMRangeAccepted[0]:
                    print(str(lineCounter) + ' fw: ' + str(fw))
                    continue
                HalfMaxes.append(fw)
                allX.append(dgdx)
                gausses.append(self.Gauss(x_g,*popt))
        count = 0
        for i in gausses:
            for k in range(len(i)):
                if i[k] < 0:
                    gausses[count][k] = -i[k]
            count +=1

        meanCurve = np.mean(gausses, axis=0)
        return HalfMaxes, gausses, meanCurve

    def countPeaks(self, line):
        maxVal = np.max(line)
        

    def getLineImage(self, data):

        m = self.get_direction(data)
        P_start, P_end = self._getPoints_new(data, m, skip = 5)
        lines = self._getLines(P_start, P_end, skip=0) #TODO: Test with and without skip, maybe later implement, that user can choose skip in GUI
        data = self._drawLinesToData(lines, data)

        return data

    def getDataOnLine(self, data, line):
        values = []
        
        for p in line:
            values.append(float(data[p[0], p[1]]))
        return values

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def Gauss(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    def FWHM(self, maximum, mean_val, sigma):
        HM = maximum/2
        FW = (2*np.sqrt(2*np.log(2)))*abs(sigma)
        #plt.hlines(HM, mean_val-FW/2, mean_val+FW/2, 'k', linestyles='dotted')
        # plt.arrow(mean_val, HM, -FW/2, 0, head_width=.02, head_length=0.1,
        #           fc='k', ec='k', ls = '-', alpha = 0.4,
        #           length_includes_head=True)
        # plt.arrow(mean_val, HM, +FW/2, 0, head_width=.02, head_length=0.1,
        #           fc='k', ec='k', ls = '-', alpha = 0.4,
        #           length_includes_head=True)
        # plt.text(mean_val-FW/4, HM-0.05, '$FWHM$ = \n' + str(round(FW*1000,2)) + ' nm')
        return FW*1000 #in nm

    def find_line_peaks(self, line_data, num_peaks, height):
        pos_peaks, _ = find_peaks((line_data), height=height, distance = 30) 
        neg_peaks, _ = find_peaks(-(line_data), height=height, distance = 30) 
        pos = []
        neg = []
        indizes_plus = []
        indizes_minus = []
        peaks = np.concatenate((pos_peaks, neg_peaks))
        vals = []
        for i in peaks:
            if line_data[i] < 0:
                vals.append(-line_data[i])
            else:
                vals.append(line_data[i])
        #print(sorted(zip(vals, peaks), reverse=False))
        highest = sorted(zip(vals, peaks), reverse=True)[:num_peaks]

        return highest

    def detectEdges(self, data):
        maxSize = max(np.shape(data))

        for idx, x in np.ndenumerate(data):
            if x<125:
                data[idx] = 0
            elif x > 125:
                data[idx] = 250
        edges = cv2.Canny(data,50,100,apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=20, # Min number of votes for valid line
            minLineLength=maxSize*.7, # Min allowed length of line
            maxLineGap=20 # Max allowed gap between line for joining them
            )
        lines_list = []
        deltaX = []
        deltaY = []
        for idx, x in np.ndenumerate(data):
            data[idx] = 255
        for points in lines:
        # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(data,(x1,y1),(x2,y2),(0,130,0),1)
            # Maintain a simples lookup list for points
            lines_list.append([(x1,y1),(x2,y2)])

        return data

    def get_direction(self, data):
        data = (255*(data - np.min(data))/np.ptp(data)).astype(int)  
        data = np.uint8(data)
        maxSize = max(np.shape(data))

        for idx, x in np.ndenumerate(data):
            if x<125:
                data[idx] = 0
            elif x > 125:
                data[idx] = 250
        edges = cv2.Canny(data,50,100,apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=20, # Min number of votes for valid line
            minLineLength=maxSize*.7, # Min allowed length of line
            maxLineGap=20 # Max allowed gap between line for joining them
            )
        lines_list = []
        deltaX = []
        deltaY = []
        for idx, x in np.ndenumerate(data):
            data[idx] = 255
        for points in lines:
        # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(data,(x1,y1),(x2,y2),(0,130,0),1)
            # Maintain a simples lookup list for points
            lines_list.append([(x1,y1),(x2,y2)])
            deltaX.append(x2-x1)
            deltaY.append(y2-y1)
  
        # get median slope
        mx = np.mean(deltaX)
        my = np.mean(deltaY)
        # rotate 90 degrees
        mx = mx
        my = -my

        if mx == 0:
            m = 'nan'
        else:
            m = my/mx
        return m

    def _drawLinesToData(self, lines, data):
        if len(np.shape(data)) == 2:
            x,y = np.shape(data)
        elif len(np.shape(data)) == 3:
            x,y,z = np.shape(data)
        else:
            print('ERROR')
        color = (np.amax(data)+np.amin(data))/2
        for line in lines:
            for point in line:
                if point[0] >= x:
                    continue
                elif point[1] >= y:
                    continue
                else:
                    data[point] = color
        return data

    def _getPoints_new(self, data, m, skip=5):
        if len(np.shape(data)) == 2:
            x,y = np.shape(data)
        elif len(np.shape(data)) == 3:
            x,y,z = np.shape(data)
        else:
            print('ERROR')
        x=x-1
        y=y-1
        P_start = []
        P_end = []
        skipCheck = 0
        if m>0:
            if m<1:
                #iterate over left side of image
                for yi in range(y):                
                    
                    if skipCheck in range(skip-1):
                        skipCheck += 1
                        continue
                    skipCheck=0
                    
                    yP = int(m*x+yi) #get point of intersection (PoS) with right edge of image

                    if yP > y-1: #if endPoS is outside of image
                        #get PoS w/ upper edge
                        xP = int((y-yi)/m)

                        P_start.append([0, yi])
                        P_end.append([xP, y])

                    else:
                        P_start.append([0, yi])
                        P_end.append([x, yP])

                yRe = int(P_end[0][1])

                #iterate over rest
                for yi in range(yRe):
                    
                    if skipCheck in range(skip-1):
                        skipCheck += 1
                        continue
                    skipCheck=0

                    xP=-int((-yi)/m)

                    P_start.append([xP, 0])
                    P_end.append([x , yRe-yi])

            else: #use upper and lower boarders

                for xi in range(x):
                
                    if skipCheck in range(skip-1):
                        skipCheck += 1
                        continue
                    skipCheck=0

                    tP = int(-m*xi) #get intersept (outside of image)
                    xP= int((y+tP)/m)

                    if xP>x-1:

                        #get PoS w/ right img boarder
                        yP = int(m*x+tP)

                        P_start.append([xi, 0])
                        P_end.append([x, yP])

                    else:
                        P_start.append([xi, 0])
                        P_end.append([xP, y])

                xRe = int(P_end[0][0])

                for xi in range(xRe):

                    if skipCheck in range(skip-1):
                        skipCheck += 1
                        continue
                    skipCheck=0

                    tP = int(-m*-xi)
                    
                    #TODO: implement if clause when start is not on right axis

                    P_start.append([0, tP])
                    P_end.append([xRe-xi, y])

        #-----------

        elif m < 0:#TODO

            if m>-1:
                #iterate (reversed) over left side of image
                for yi in reversed(range(y)):

                    if skipCheck in range(skip-1):
                        skipCheck += 1
                        continue
                    skipCheck=0

                    yP = int(m*x+yi) #get point of intersection (PoS) with right edge of image

                    if yP < 0: #if PoS is outside of image
                        #get PoS bottom & left edge
                        xP = int(-yi/m)
                        P_start.append([0,yi])
                        P_end.append([xP, 0])

                    else:
                        P_start.append([0, yi]) #PoS left
                        P_end.append([x, yP]) #Pos right

                yRe = y - int(P_end[0][1])

                #iterate over rest (PoS top, right)
                for yi in range(yRe):

                    if skipCheck in range(1, skip):
                        skipCheck += 1
                        continue
                    skipCheck=1

                    tP = int(y+yi)

                    xP= int((y-tP)/m)

                    P_start.append([xP, y])
                    P_end.append([x , yRe+yi])

            else:

                #iterate (reversed) over bottom side of image
                for xi in reversed(range(x)):
                    if skipCheck in range(skip-1):
                        skipCheck += 1
                        continue
                    skipCheck=0
                    
                    #get PoS w/ top
                    tP = int(-m*xi)
                    xP = int((y-tP)/m)

                    if xP <0:

                        #get PoS w/ left
                        P_start.append([xi, 0])
                        P_end.append([0, tP])

                    else:

                        #use PoS
                        P_start.append([xi,0])
                        P_end.append([xP, y])

                xRe = int(P_end[0][0])

                #iterate over rest ()

                for xi in range(x-xRe):
                
                    if skipCheck in range(1, skip):
                        skipCheck += 1
                        continue
                    skipCheck=1
                    
                    tP = -m*(x+xi)
                    yP = int(m*x+tP)
                    xP = int((y-tP)/m)
                    
                    P_start.append([x,yP])
                    P_end.append([xP, y])
                    

        elif m==0:

            for yi in range(y):

                if skipCheck in range(skip-1):
                    skipCheck += 1
                    continue
                skipCheck=0
                    
                P_start.append([0,yi])
                P_end.append([x, yi])

        else:

            for xi in range(x):

                if skipCheck in range(skip-1):
                    skipCheck += 1
                    continue
                skipCheck=0

                P_start.append([xi,0])

                P_end.append([xi, y])

        return P_start, P_end

    def _getPoints_old(self, data, m):
        x,y,z = np.shape(data)
        P_start = []
        P_end = []

        if m>0:
            #iterate over left side of image
            for yi in range(y):
                P_start.append([0, yi])
                xP = (y - yi)/m #get point of intersection (PoS) with top edge of image
                if xP > x-1: #if PoS is outside of image
                    yP = m*x+yi #use PoS with right edge of image
                    P_end.append([x, int(yP)])
                else:
                    P_end.append([int(xP), y])
            #iterate over bottom side of image
            for xi in range(x):
                P_start.append([x,0])
                t = -m*xi #get intesept (outside of image)
                yP = m*x+t #get PoS with right side of image
                if yP > y-1: #if PoS is outside of image
                    xP = (y-t)/m #use PoS with top edge of image
                    P_end.append([int(xP), y])
                else:
                    P_end.append([x, int(yP)])
        elif m<0:
            #if abs(m)<1:
            #iterate over left side of image
            for yi in range(y):
                P_start.append([0,yi])
                xP = -yi/m #get PoS with bottom edge of image
                if xP>x-1: #if PoS outside of image 
                    yP = m*x+yi #get PoS with right edge of image
                    P_end.append([x, int(yP)])
                else:
                    P_end.append([int(xP), 0])
            #iterate over top side of image
            for xi in range(x):
                P_start.append([xi, y])
                tP = y-m*xi #get intersept (outside of image)
                xP = -tP/m #get PoS with bottom edge of image
                if xP>x-1: #if PoS outside image
                    yP=m*x+tP #use PoS with right edge of image
                    P_end.append([x, int(yP)])
                else:
                    P_end.append([int(xP), 0])
        elif m==0:
            for yi in range(y):
                P_start.append([0,yi])
                P_end.append([x, yi])
        else:
            for xi in range(x):
                P_start.append([xi,0])
                P_end.append([xi, y])
        return P_start, P_end


    def _getLines(self, Pstart, Pend, skip=0):
        lines = []
        i=-1
        for p1, p2 in zip(Pstart, Pend):
            i+=1
            if i == skip:
                lines.append(self._getLine(p1,p2))
                i=-1
        return lines

    def _getLine(self, start, end):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end

        >>> points1 = get_line((0, 0), (3, 4))
        >>> points2 = get_line((3, 4), (0, 0))
        >>> assert(set(points1) == set(points2))
        >>> print points1
        [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
        >>> print points2
        [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)

        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []

        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)

            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()

        return points