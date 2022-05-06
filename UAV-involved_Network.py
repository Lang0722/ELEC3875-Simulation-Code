import math
import numpy
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

r = 800  # Radius of MC
r_SC = 30  # Radius of SC
n = 50  # number of UEs
n_SC = 100  # number of SCs

# number of UAVs
n_UAV = 3
UAV_record = [0, 0, 0]
UAV_Angle = [0, np.pi * 2 / 3, np.pi * 4 / 3]

# HO record
HO_info = [[], [], [], []]
HO_result = []

class HOrecord:
    def __init__(self, total, HOF, UHO):
        self.total = total
        self.HOF = HOF
        self.UHO = UHO

    def updatetotal(self):
        self.total = self.total + 1

    def updateHOF(self):
        self.HOF = self.HOF + 1

    def updateUHO(self):
        self.UHO = self.UHO + 1


def initial_model(u_radial, u_angle, sc_radial, sc_angle, plotting=False):
    radii = np.zeros(n)  # the radial coordinate of the points
    angle = np.zeros(n)  # the angular coordinate of the points
    radii_SC = np.zeros(n_SC)  # the radial coordinate of the points for SC
    angle_SC = np.zeros(n_SC)  # the angular coordinate of the points for SC

    for i in range(n):
        radii[i] = r * (np.sqrt(u_radial[i]))
        angle[i] = 2 * np.pi * u_angle[i]

    for i in range(n_SC):
        radii_SC[i] = r * (np.sqrt(sc_radial[i]))
        angle_SC[i] = 2 * np.pi * sc_angle[i]

    # Generate Coordinates for UEs,SCs
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(n):
        x[i] = radii[i] * np.cos(angle[i])
        y[i] = radii[i] * np.sin(angle[i])

    x_SCs = np.zeros(n_SC)
    y_SCs = np.zeros(n_SC)
    for i in range(n_SC):
        x_SCs[i] = radii_SC[i] * np.cos(angle_SC[i])
        y_SCs[i] = radii_SC[i] * np.sin(angle_SC[i])

    # Coordinates for UAVs
    x_UAVs = np.zeros(n_UAV)
    y_UAVs = np.zeros(n_UAV)
    x_cor = [-433, 433, 0]
    y_cor = [-250, -250, 500]
    for i in range(n_UAV):
        x_UAVs[i] = x_cor[i]
        y_UAVs[i] = y_cor[i]

    if (plotting):
        fig = plt.figure(figsize=(5, 5))
        plt.plot(x, y, 'bo', markersize=1.5)  # plot UEs
        plt.plot(x_SCs, y_SCs, '*g', markersize=1.5)  # plot SCs
        plt.plot(x_UAVs, y_UAVs, '*r', markersize=2.5)  # plot SCs
        plt.plot(0, 0, '^r')
        plt.grid(True)

        plt.xlabel('X pos (m)')
        plt.ylabel('Y pos (m)')

        ax = plt.gca()
        circ = plt.Circle((0, 0), radius=r, color='r', linewidth=2, fill=False)
        ax.add_artist(circ)

        plt.xlim(-r, r)
        plt.ylim(-r, r)

        plt.show()

    return [x, y, x_SCs, y_SCs, x_UAVs, y_UAVs]



def update_distance():
    for i in range(n):
        distances[i][0] = np.sqrt(np.square(UEs_X[i]) + np.square(UEs_Y[i]))  # distance between UE and MC

    for i in range(n):  # distance between UE and SCs
        for j in range(1, n_SC + 1):
            distances[i][j] = np.sqrt(np.square(UEs_X[i] - Xsc[j - 1]) + np.square(UEs_Y[i] - Ysc[j - 1]))

    for i in range(n):
        distances[i][n_SC + 1] = np.sqrt(np.square(UEs_X[i] - Xuav[0]) + np.square(UEs_Y[i] - Yuav[0]) + np.square(50))
        distances[i][n_SC + 2] = np.sqrt(np.square(UEs_X[i] - Xuav[1]) + np.square(UEs_Y[i] - Yuav[1]) + np.square(50))
        distances[i][n_SC + 3] = np.sqrt(np.square(UEs_X[i] - Xuav[2]) + np.square(UEs_Y[i] - Yuav[2]) + np.square(50))


def update_RSRP():
    for i in range(n):  # UEs and MC
        pathloss_MC = 32.4 + 30 * np.log10(distances[i][0]) + 20 * np.log10(3.8)
        RSRP[i][0] = 46 + 14 - 8 - pathloss_MC

    for i in range(n):  # UEs and SCs
        for j in range(1, n_SC + 1):
            pathloss_SC = 32.4 + 31.9 * np.log10(distances[i][j]) + 20 * np.log10(28)
            RSRP[i][j] = 30 + 5 - 10 - pathloss_SC

    for i in range(n):
        pathloss_UAV1 = 28 + 22 * np.log10(distances[i][n_SC + 1]) + 20 * np.log10(2)
        pathloss_UAV2 = 28 + 22 * np.log10(distances[i][n_SC + 2]) + 20 * np.log10(2)
        pathloss_UAV3 = 28 + 22 * np.log10(distances[i][n_SC + 3]) + 20 * np.log10(2)
        RSRP[i][n_SC + 1] = 23 + 0 - 9 - pathloss_UAV1
        RSRP[i][n_SC + 2] = 23 + 0 - 9 - pathloss_UAV2
        RSRP[i][n_SC + 3] = 23 + 0 - 9 - pathloss_UAV3


def calc_angle(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dy / dx)
    elif x2 > x1 and y2 < y1:
        angle = 2 * math.pi - math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dy / dx)
    elif x2 < x1 and y2 > y1:
        angle = math.pi - math.atan(dy / -dx)
    return angle


def update_angle():
    for i in range(n):
        for j in range(n_SC):
            UESC_angles[i][j] = abs(calc_angle(UEs_X[i], UEs_Y[i], Xsc[j], Ysc[j]) - UEs_Angle[i])

        # UAV angle projection
        UESC_angles[i][n_SC] = abs(calc_angle(UEs_X[i], UEs_Y[i], Xuav[0], Ysc[0]) - UEs_Angle[i])
        UESC_angles[i][n_SC + 1] = abs(calc_angle(UEs_X[i], UEs_Y[i], Xuav[1], Yuav[1]) - UEs_Angle[i])
        UESC_angles[i][n_SC + 2] = abs(calc_angle(UEs_X[i], UEs_Y[i], Xuav[2], Yuav[2]) - UEs_Angle[i])


def update_SINR():
    for i in range(n):
        sum = 0
        for j in range(n_SC + 1 + n_UAV):
            sum = sum + pow(10, RSRP[i][j] / 10)

        for j in range(n_SC + 1 + n_UAV):
            SINR[i][j] = 10 * np.log10(pow(10, RSRP[i][j] / 10) / (sum - pow(10, RSRP[i][j] / 10)))


def update_ToS():
    for i in range(n):
        ToS[i] = 4 * r_SC / (np.pi * (UEs_V[i] / 3.6))


def randomMobility(X, Y, UEs_V, UEs_Angle):
    x_V = np.zeros(n)
    y_Angle = np.zeros(n)
    time = np.random.rand(n) * 1  # time for user randomly between [0,4)

    for i in range(n):
        x_V[i] = np.random.uniform(0, 60)  # randomly change the UE speed

        y_Angle[i] = np.random.uniform(0, 2 * np.pi)
        # update the speed angle
        update_x = x_V[i] * np.cos(y_Angle[i]) / 3.6 * time[i] + X[i]
        update_y = x_V[i] * np.sin(y_Angle[i]) / 3.6 * time[i] + Y[i]
        if np.square(update_x) + np.square(update_y) >= np.square(r):
            update_x = X[i] - x_V[i] * np.cos(y_Angle[i]) / 3.6 * time[i]
            update_y = Y[i] - x_V[i] * np.sin(y_Angle[i]) / 3.6 * time[i]

        # update the value
        X[i] = update_x
        Y[i] = update_y
        UEs_V[i] = x_V[i]
        UEs_Angle[i] = y_Angle[i]

    # UAV moving
    for i in range(n_UAV):
        t = 1
        Xuav[i] = 10 * np.cos(UAV_Angle[i]) * t + Xuav[i]
        Yuav[i] = 10 * np.sin(UAV_Angle[i]) * t + Yuav[i]
        UAV_record[i] += 10 * t
        if UAV_record[i] >= 866:
            UAV_record[i] = 0
            UAV_Angle[i] = UAV_Angle[i] + 2 * np.pi / 3
            if UAV_Angle[i] > 4 * np.pi / 3:
                UAV_Angle[i] = 0


def renewData():
    randomMobility(UEs_X, UEs_Y, UEs_V, UEs_Angle)
    update_distance()
    update_RSRP()
    update_angle()
    update_SINR()
    update_ToS()

def conventional_HO():
    index1 = np.zeros(n)
    flag = np.zeros(n)
    target = np.zeros(n)
    ToS_real = np.zeros((n, n_SC))
    UEs_status = np.ones(n)

    for i in range(n):
        index1[i] = numpy.argmax(RSRP[i])
        if index1[i] != 0:
            record.updatetotal()
            flag[i] = 1
            target[i] = index1[i]
            # print("UE", i, " change to ", index1[i])

    renewData()

    for i in range(n):
        if flag[i] == 1:
            if SINR[i][int(target[i])] < 5:
                record.updateHOF()


def proposed_HO():
    flag = np.zeros(n)
    score = np.zeros((n, n_SC))
    target = np.zeros(n)
    flag2 = np.zeros(n)  # for handover
    UEs_status = np.ones(n)  # for handover status
    UAV_flag = np.zeros(n)

    # set Vth
    for i in range(n):
        if UEs_V[i] <= 30:
            flag[i] = 1

    for i in range(n):
        if flag[i] != 0:
            candidates = []
            for j in range(n_SC):
                # ToS weight
                if ToS[i] > 2:
                    score[i][j] += 0.2
                else:
                    score[i][j] += 0.2 * ToS[i]

                # Speed angle
                if UESC_angles[i][j] <= np.pi * (1 / 3):
                    score[i][j] += 0.3
                elif np.pi * (1 / 3) <= UESC_angles[i][j] <= np.pi * (1 / 2):
                    score[i][j] += 0.3 * (np.pi / 36 * np.square(UESC_angles[i][j] - 0.5 * np.pi))

                # distance
                if distances[i][j + 1] <= 30:
                    score[i][j] += 0.5
                elif 30 < distances[i][j + 1] <= 100:
                    score[i][j] += 0.5 * (-1 / 70 * distances[i][j + 1] + 1.42)

                if score[i][j] >= 0.7:
                    candidates.append(j)

            if len(candidates) == 0:
                continue
            else:
                NCL_number = len(candidates)
                NCL_RSRP = np.zeros(NCL_number)
                for k in range(NCL_number):
                    NCL_RSRP[k] = RSRP[i][candidates[k] + 1]

                max_index = candidates[np.argmax(NCL_RSRP)]
                if RSRP[i][max_index + 1] > RSRP[i][0]:
                    if SINR[i][max_index + 1] >= 5:
                        record.updatetotal()
                        # log this UE's HO
                        flag2[i] = 1
                        target[i] = max_index

    renewData()

    # HOF
    for i in range(n):
        if flag2[i] == 1:
            if SINR[i][int(target[i]) + 1] < 5:
                record.updateHOF()

                if (UEs_status[i] != 0):
                    # log this HO to dataset
                    HO_info[0].append(RSRP[i][int(target[i]) + 1])
                    HO_info[1].append(distances[i][int(target[i]) + 1])
                    HO_info[2].append(ToS[i])
                    HO_info[3].append(UESC_angles[i][int(target[i])])

                    # update the status for HO_info
                    HO_result.append(0)
            elif UEs_status[i] != 0:
                # log this HO to dataset
                HO_info[0].append(RSRP[i][int(target[i]) + 1])
                HO_info[1].append(distances[i][int(target[i]) + 1])
                HO_info[2].append(ToS[i])
                HO_info[3].append(UESC_angles[i][int(target[i])])

                # update the status for HO_info
                HO_result.append(1)


# start process
total = 0
HOF = 0
UHO = 0
record = HOrecord(total, HOF, UHO)


# set up environment
np.random.seed(17)
# initial the UEs and SCs position
sc_radial = np.random.uniform(0.0, 1.0, n_SC)  # generate another n uniformly distributed points for small cells
sc_angle = np.random.uniform(0.0, 1.0, n_SC)  # generate another n uniformly distributed points for small cells


u_radial = np.random.uniform(0.0, 1.0, n)  # generate n uniformly distributed points for UE
u_angle = np.random.uniform(0.0, 1.0, n)  # generate another n uniformly distributed points for UE

UEs_V = np.random.uniform(0, 60, n)  # generate n uniformly speed value for UE between [0,60) km/h
UEs_Angle = np.random.uniform(0, 2 * np.pi, n)  # initial UE speed angle between [0,2pi)

RSRP = np.zeros((n, n_SC + 1 + n_UAV))
distances = np.zeros((n, n_SC + 1 + n_UAV))
UESC_angles = np.zeros((n, n_SC + n_UAV))
SINR = np.zeros((n, n_SC + 1 + n_UAV))
ToS = np.zeros(n)

# Initial the data for UEs and SCs
([UEs_X, UEs_Y, Xsc, Ysc, Xuav, Yuav]) = initial_model(u_radial, u_angle, sc_radial, sc_angle, plotting=False)
randomMobility(UEs_X, UEs_Y, UEs_V, UEs_Angle)
update_distance()
update_RSRP()
update_angle()
update_SINR()
update_ToS()

for i in range(360):
    proposed_HO()
    # conventional_HO()

print("Total HO: ", record.total/10)
print("HOF: ", record.HOF/10)
print("UHO: ", record.UHO/10)
print("HOF pro: ", record.HOF/record.total)
print("finish")

# # SVM test
# collect_data = np.array(HO_info).T
# collect_result = np.array(HO_result)
#
# train_data, test_data = train_test_split(preprocessing.scale(collect_data), random_state=1, train_size=0.7,
#                                          test_size=0.3)
# train_label, test_label = train_test_split(collect_result, random_state=1, train_size=0.7, test_size=0.3)
#
# classifier = svm.SVC(C=25, kernel='rbf', gamma='auto')
# classifier.fit(train_data, train_label.ravel())
#
# pre_train = classifier.predict(train_data)
# pre_test = classifier.predict(test_data)
#
# print("sample", len(collect_data))
# print("train", accuracy_score(train_label, pre_train))
# print("test", accuracy_score(test_label, pre_test))
#
# print('finish')
