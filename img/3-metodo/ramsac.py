    def fit_lines_to_endpoints(self, end_pts, thresh=1., max_error=1.):
        X = end_pts[:, 0].reshape(-1, 1)
        Y = end_pts[:, 1].reshape(-1, 1)

        Lines = []
        Coefs = []
        linReg = pr.Ransac(e=0.2)
        M = np.hstack([X, Y])

        N, _ = M.shape
        Idx = np.arange(N)

        [coefs, Error] = linReg.fitRansac(M, thrFact=thresh)
        if Error == None:
            return None

        iX = X[Idx[linReg.inliersIdx], 0]
        iY = Y[Idx[linReg.inliersIdx], 0]

        Coefs.append([coefs, Error, iX, iY])

        idxMask = np.ones(len(X)).astype('bool')
        idxMask[linReg.inliersIdx] = False

        while (Error < max_error):
            sX = end_pts[idxMask, 0].reshape(-1, 1)
            sy = end_pts[idxMask, 1].reshape(-1, 1)
            Idx = np.arange(N)[idxMask]

            M = np.hstack([sX, sy])
            n, _ = M.shape
            if n < 5:
                break

            [coefs, Error] = linReg.fitRansac(M, thrFact=1.)
            if Error == None:
                break
            iX = X[Idx[linReg.inliersIdx], 0]
            iY = Y[Idx[linReg.inliersIdx], 0]
            Coefs.append([coefs, Error, iX, iY])
            idxMask[Idx[linReg.inliersIdx]] = False

        end_pts_line_eqs = []
        end_pts_info = []
        end_pts_lines = np.zeros((len(Coefs), 1, 4))
        idx = 0
        for C in Coefs:
            iX = C[2]
            iY = C[3]

            if min(iX) != max(iX):
                indices = sorted(range(len(iX)), key=lambda index: X[index])
                iX = iX[indices]
                iY = iY[indices]
            elif min(iY) != max(iY):
                indices = sorted(range(len(iY)), key=lambda index: Y[index])
                iX = iX[indices]
                iY = iY[indices]
            else:
                print("fit_lines_to_endpoints:I should throw an exception!")
            end_pts_line_eqs.append(C[0][:3])
            end_pts_lines[idx, 0, :] = [iX[0], iY[0], iX[-1], iY[-1]]
            end_pts_info.append([C[1], iX, iY])
            idx += 1

        return end_pts_line_eqs, end_pts_lines, end_pts_info