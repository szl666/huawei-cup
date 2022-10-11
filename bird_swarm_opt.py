from numpy import *


class Bird_swarm_opt:
    def __init__(self, dim=10, min_value=-6, max_value=6, X_og=[]):
        self.X_og = X_og
        self.dim = dim
        self.min_value = min_value
        self.max_value = max_value
        self.lb, self.ub = self.get_lb_ub()

    def get_lb_ub(self):
        lb = [self.min_value] * self.dim
        ub = [self.max_value] * self.dim
        lb = expand_dims(lb, axis=0)
        ub = expand_dims(ub, axis=0)
        return lb, ub

    def cal_RMSE_infected(self, X_11_dis):
        RMSE = 0
        Infect = 0
        for i in range(len(hyper)):
            #print(district_name[i])
            model = get_model(eval(hyper[i]), np.array(X_dis[i]),
                              np.array(Y_dis[i]))
            X_dis_new = np.array(X_11_dis[i])
            for j in range(len(X_dis_new)):
                if j > 0:
                    X_dis_new[j][4] = model.predict(X_dis_new)[j - 1]
            Infect += model.predict(X_dis_new)
            RMSE += mean_squared_error(Y_dis[i][15:21],
                                       model.predict(X_dis_new))**(1 / 2) / 6
        #print(model)
        return (RMSE, Infect)

    def get_infect(self, x):
        X = generate_X(x, self.X_og)
        RMSE, Infected = self.cal_RMSE_infected(X)
        return sum(Infected) / 6

    def randiTabu(self, minm, maxm, tabu, dim):
        value = ones((dim, 1)) * maxm * 2
        num = 1
        while (num <= dim):
            temp = random.randint(minm, maxm)
            findi = [
                index for (index, values) in enumerate(value) if values != temp
            ]
            if (len(findi) == dim and temp != tabu):
                value[0][num - 1] = temp
                num += 1
        return value

    def Bounds(self, s, lb, ub):
        temp = s
        I = [
            index for (index, values) in enumerate(temp)
            if values < lb[0][index]
        ]
        for indexlb in I:
            temp[indexlb] = lb[0][indexlb]
        J = [
            index for (index, values) in enumerate(temp)
            if values > ub[0][index]
        ]
        for indexub in J:
            temp[indexub] = ub[0][indexub]
        return s

    def search(self, pop=30, M=100, FQ=10, c1=0.15, c2=0.15, a1=1, a2=1):
        #############################################################################
        #     Initialization
        x = zeros([pop, self.dim])
        fit = zeros([pop, 1])
        for i in range(0, pop):
            x[i, :] = self.lb[0] + (self.ub[0] - self.lb[0]) * random.rand(
                1, self.dim)
            x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
            fit[i] = self.get_infect(x[i, :])
        pFit = fit.copy()
        pX = x.copy()
        fMin = float(min(pFit))
        print(fMin)
        bestIndex = argmin(pFit)
        bestX = pX[bestIndex, :]
        print(bestX)
        b2 = zeros([M, 1])
        b2[0] = fMin
        #     Start the iteration.
        for index in range(0, M):
            print(f'Iteration {index}')
            prob = random.rand(pop, 1) * 0.2 + 0.8
            if (mod(index, FQ) != 0):
                ###############################################################################
                #     Birds forage for food or keep vigilance
                sumPfit = pFit.sum()
                meanP = mean(pX)
                realmin = finfo(float).tiny
                for i in range(0, pop):
                    if random.rand() < float(prob[i]):
                        x[i, :] = x[i, :] + c1 * random.rand() * (bestX - x[
                            i, :]) + c2 * random.rand() * (pX[i, :] - x[i, :])
                    else:
                        person = int(self.randiTabu(1, pop, i, 1)[0])
                        x[i, :] = x[i, :] + random.rand() * (
                            meanP - x[i, :]) * a1 * exp(
                                -pFit[i] / (sumPfit + realmin) * pop) + a2 * (
                                    random.rand() * 2 - 1
                                ) * (pX[person, :] - x[i, :]) * exp(
                                    -(pFit[person] - pFit[i]) /
                                    (abs(pFit[person] - pFit[i]) + realmin) *
                                    pFit[person] / (sumPfit + realmin) * pop)
    #                 print(x[i,:])
                    x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    #                 print(x[i,:])
                    fit[i] = self.get_infect(x[i, :])
    ###############################################################################
            else:
                FL = random.rand(pop, 1) * 0.4 + 0.5
                ###############################################################################
                #     Divide the bird swarm into two parts: producers and scroungers
                minIndex = argmin(pFit)
                maxIndex = argmax(pFit)
                choose = 0
                if (minIndex < 0.5 * pop and maxIndex < 0.5 * pop):
                    choose = 1
                elif (minIndex > 0.5 * pop and maxIndex < 0.5 * pop):
                    choose = 2
                elif (minIndex < 0.5 * pop and maxIndex > 0.5 * pop):
                    choose = 3
                elif (minIndex > 0.5 * pop and maxIndex > 0.5 * pop):
                    choose = 4
    ###############################################################################
                if choose < 3:
                    for i in range(int(pop / 2 + 1) - 1, pop):
                        x[i, :] = x[i, :] * (1 + random.randn())
                        #                     print(x[i,:])
                        x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                        fit[i] = self.get_infect(x[i, :])
                    if choose == 1:
                        x[minIndex, :] = x[minIndex, :] * (1 + random.randn())
                        x[minIndex, :] = self.Bounds(x[minIndex, :], self.lb,
                                                     self.ub)
                        fit[minIndex] = self.get_infect(x[minIndex, :])
                    for i in range(0, int(0.5 * pop)):
                        if choose == 2 or minIndex != i:
                            # print(type(pop))
                            person = random.randint(0.5 * pop + 1, pop)
                            # print(person)
                            x[i, :] = x[i, :] + (pX[person, :] -
                                                 x[i, :]) * FL[i]
                            x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                            fit[i] = self.get_infect(x[i, :])
                else:
                    for i in range(0, int(0.5 * pop)):
                        x[i, :] = x[i, :] * (1 + random.randn())
                        x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                        fit[i] = self.get_infect(x[i, :])
                    if choose == 4:
                        x[minIndex, :] = x[minIndex, :] * (1 + random.randn())
                        x[minIndex, :] = self.Bounds(x[minIndex, :], self.lb,
                                                     self.ub)
                        fit[minIndex] = self.get_infect(x[minIndex, :])
                    for i in range(int(0.5 * pop), pop):
                        if choose == 3 or minIndex != i:
                            # print(type(pop))
                            person = random.randint(1, 0.5 * pop + 1)
                            # print(person)
                            x[i, :] = x[i, :] + (pX[person, :] -
                                                 x[i, :]) * FL[i]
                            x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                            fit[i] = self.get_infect(x[i, :])

    ###############################################################################
    #     Update the individual's best fitness value and the global best one
            for i in range(0, pop):
                if (fit[i] < pFit[i]):
                    pFit[i] = fit[i]
                    pX[i, :] = x[i, :]
                if (pFit[i] < fMin):
                    fMin = pFit[i]
            fMin = float(min(pFit))
            print(fMin)
            bestIndex = argmin(pFit)
            bestX = pX[bestIndex, :]
            b2[index] = fMin
            save("fMin.npy", fMin)
            save("bestIndex.npy", bestIndex)
            save("bestX.npy", bestX)
            save("b2.npy", b2)
        # print(fMin)
        return fMin, bestIndex, bestX, b2
