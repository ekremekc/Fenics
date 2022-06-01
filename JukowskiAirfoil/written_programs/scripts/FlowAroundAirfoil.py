import dolfin as dolf



class FlowAroundAirfoil:
    """
        Calculating the flow around a given airfoil.
        Created fields:
            u, p: The velocity and pressure fields of the flow.
                (uFull, u_x, u_y are also calculated)
            eigenvalues: List of (sReal, sImag) eigenvalue pairs. Only calculated if either 
                findEigenvalues() or getMaxEigenvalue() is called.
            eigenflows: List of (wReal, wImag) eigenflow pairs. Only calculated if either 
                findEigenvalues() or getMaxEigenvalue() is called.
        Callable functions:
            saveSolution(): Save the calculated flow to a given file.
            dragForceDirect():  The drag force on the airfoil calculated by a surface 
                integral on the airfoil.
            dragForceGauss(): The drag force on the airfoil calculated using Gauss' law.
            forceDerivative(): The derivative of the force on the airfoil with respect 
                to shape changes.
            costLD(): The Lift-Drag force ratio on the airfoil.
            costLDDerivative(): The derivative of the Lift-Drag force ratio on the airfoil 
                with respect to shape changes.
            findEigenvalues(): Find the perturbation fields and their eigenvalues.
            getMaxEigenvalue(): Find the most unstable eigenvalue.
            eigenvalueDerivative(): Find the derivative of an eigenvalue with respect to 
                shape changes.
    """
    
    def __init__(self, airfoil, v_in, mu, rho, onlyFlow=False):
        """
            Initialising arguments:
                airfoil: the investigeated airfoil
                v_in: inflow velocity
                mu: dynamic viscosity
                rho: density of fluid
            Optional argument:
                onlyFlow: If True only the flow around the airfoil is calculated.
        """
        self.airfoil = airfoil
        self.v_in = v_in
        self.mu = mu
        self.rho = rho
        
        # If only the flow is needed
        if(onlyFlow):
            self.solveNSFlow()
            return None
        
        self.solveNSFlow()
        self.smoothPhi()
    
    # Solving the Stokes problem
    def solveStokesFlow(self):
        """
            The solution to the steady Stokes flow problem.
            0 = grad(p) - mu*laplace(u)
            0 = div(u)
        """
        mu = self.mu
        
        # Define function space
        # (u_x, u_y, p)
        P1 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),1)  # for pressure
        P2 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),2)  # for a velocity component
        W = dolf.FunctionSpace(self.airfoil.mesh, dolf.MixedElement([P2,P2,P1]))
        
        # Trial and test functions
        uFull = dolf.TrialFunction(W)
        vFull = dolf.TestFunction(W)
        (u_x, u_y, p) = dolf.split(uFull)
        u = dolf.as_vector([u_x,u_y])
        (v_x, v_y, q) = dolf.split(vFull)
        v = dolf.as_vector([v_x,v_y])
        
        # Setting the boundary conditions
        # Inflow velocity
        leftBoundary = self.airfoil.leftBoundary
        class LeftBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return leftBoundary(x, on_boundary)
        bcuInflowX = dolf.DirichletBC(W.sub(0), self.v_in, LeftBoundarySubDomain())
        bcuInflowY = dolf.DirichletBC(W.sub(1), 0, LeftBoundarySubDomain())
        
        # Zero y velocity at the bottom and top
        bottomTopBoundary = self.airfoil.bottomTopBoundary
        class BottomTopBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return bottomTopBoundary(x, on_boundary)
        bcuBottomTopY = dolf.DirichletBC(W.sub(1), 0, BottomTopBoundarySubDomain())
        
        # No slip on the airfoil
        airfoilBoundary = self.airfoil.airfoilBoundary
        class AirfoilBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return airfoilBoundary(x, on_boundary)
        bcuAirfoilX = dolf.DirichletBC(W.sub(0), 0, AirfoilBoundarySubDomain())
        bcuAirfoilY = dolf.DirichletBC(W.sub(1), 0, AirfoilBoundarySubDomain())
        
        # Collecting the boundary conditions together
        bcu = [bcuInflowX, bcuInflowY, bcuBottomTopY, bcuAirfoilX, bcuAirfoilY]
        
        # Define varational problem
        # The boundary terms sum to zero
        F = -dolf.div(v)*p*dolf.dx + \
            mu*dolf.inner(dolf.grad(v),dolf.grad(u))*dolf.dx + \
            q*dolf.div(u)*dolf.dx
        
        # Solving the variational problem
        uFull = dolf.Function(W)
        dolf.solve(dolf.lhs(F) == dolf.rhs(F), uFull, bcu)
        
        return uFull
    
    # Solving the Navier-Stokes problem
    def solveNSFlow(self, uInitial=None):
        """
            Solution to the steady Navier-Stokes problem.
            Optional argument:
                uInitial: The initial guess for the flow. This needs to satisfy the boundary
                          conditions of the flow with the exception of the inflow velocity.
            The L2 tolerance is 1E-14*norm(uInitial)
            The following variables are created:
                uFull
                u_x,u_y,p
                u
            rho*u*nabla_grad(u) + grad(p) - mu*laplace(u) = 0
            div(u) = 0
            For low Reynolds numbers the solution to the steady Stokes problem is used as an initial guess.
            For higher Reynolds numbers the initial guess is the solution to the steady Navier-Stokes 
            problem at lower Reynolds number.
        """
        mu = self.mu
        rho = self.rho
        
        if(uInitial == None):
            # An approximate Reynolds number
            reynolds = self.airfoil.deltay*rho*self.v_in/mu
            reynoldsLimit = 40.
            if(reynolds < reynoldsLimit):
                uInitial = self.solveStokesFlow()
            else:
                initialFlow = FlowAroundAirfoil(self.airfoil, 0.5*self.v_in, self.mu, self.rho, True)
                uInitial = initialFlow.uFull
                
        
        iterTol = 1E-14*dolf.norm(uInitial, 'L2')
        # Define function space
        # (u_x, u_y, p)
        P1 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),1)  # for pressure
        P2 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),2)  # for a velocity component
        W = dolf.FunctionSpace(self.airfoil.mesh, dolf.MixedElement([P2,P2,P1]))
        
        # Original, trial and test functions
        uFull = dolf.interpolate(uInitial,W)
        (u_x, u_y, p) = uFull.split()
        u = dolf.as_vector([u_x,u_y])
        
        # Setting the boundary conditions
        # Inflow velocity
        leftBoundary = self.airfoil.leftBoundary
        class LeftBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return leftBoundary(x, on_boundary)
        # This boundary condition should be updated with the correction term during every iteration
        bcwInflowX = dolf.DirichletBC(W.sub(0), 0, LeftBoundarySubDomain())
        bcwInflowY = dolf.DirichletBC(W.sub(1), 0, LeftBoundarySubDomain())
        
        # Zero y velocity at the bottom and top
        bottomTopBoundary = self.airfoil.bottomTopBoundary
        class BottomTopBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return bottomTopBoundary(x, on_boundary)
        bcwBottomTopY = dolf.DirichletBC(W.sub(1), 0, BottomTopBoundarySubDomain())
        
        # No slip at the airfoil
        airfoilBoundary = self.airfoil.airfoilBoundary
        class AirfoilBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return airfoilBoundary(x, on_boundary)
        bcwAirfoilX = dolf.DirichletBC(W.sub(0), 0, AirfoilBoundarySubDomain())
        bcwAirfoilY = dolf.DirichletBC(W.sub(1), 0, AirfoilBoundarySubDomain())
        
        # Collecting the boundary conditions
        # It is important to have bcw[0]=bcwInflowX as it needs to be updated at every iteration
        bcw = [bcwInflowX, bcwInflowY, bcwBottomTopY, bcwAirfoilX, bcwAirfoilY]
        
        cycleNum = 0
        L2Norm = 1.+iterTol
        print("Solving Navier-Stokes iteratively.")
        while(L2Norm > iterTol):
            # Original, trial and test functions
            vFull = dolf.TestFunction(W)
            wFull = dolf.TrialFunction(W)
            (v_x, v_y, q) = dolf.split(vFull)
            (w_x, w_y, r) = dolf.split(wFull)
            v = dolf.as_vector([v_x,v_y])
            w = dolf.as_vector([w_x,w_y])
            
            # Update inflow velocity correction term
            u_x_correction = dolf.project(self.v_in-u_x, dolf.FunctionSpace(self.airfoil.mesh, P2))
            bcw[0] = dolf.DirichletBC(W.sub(0), u_x_correction, LeftBoundarySubDomain())
            
            # Define varational problem
            # The boundary terms sum to zero
            FOriginal = rho*dolf.dot(v,dolf.dot(u,dolf.nabla_grad(u)))*dolf.dx + \
                        -dolf.div(v)*p*dolf.dx + \
                        mu*dolf.inner(dolf.grad(v),dolf.grad(u))*dolf.dx + \
                        q*dolf.div(u)*dolf.dx
            FDer = rho*dolf.dot(v,dolf.dot(w,dolf.nabla_grad(u)))*dolf.dx + \
                rho*dolf.dot(v,dolf.dot(u,dolf.nabla_grad(w)))*dolf.dx + \
                -dolf.div(v)*r*dolf.dx + \
                mu*dolf.inner(dolf.grad(v),dolf.grad(w))*dolf.dx + \
                q*dolf.div(w)*dolf.dx
            F = FOriginal + FDer
            
            # Calculating the norm of the original term
            b=dolf.PETScVector()
            dolf.assemble(FOriginal,tensor=b)
            for bc in bcw:
                bc.apply(b)
            L2Norm = dolf.norm(b, 'L2')
            
            # Solving the problem
            wFull = dolf.Function(W)
            dolf.solve(dolf.lhs(F) == dolf.rhs(F), wFull, bcw)
            
            # Updating the functions
            uFull.vector()[:] += wFull.vector()[:]
            (u_x, u_y, p) = uFull.split()
            u = dolf.as_vector([u_x,u_y])
            # Updating cycle number
            cycleNum += 1
        print("Navier-Stokes solved in %i iterations."%(cycleNum))
        
        self.uFull = uFull
        self.u_x, self.u_y, self.p = uFull.split()
        self.u = dolf.as_vector([u_x,u_y])
        
        return None
    
    def saveSolution(self, (uFile, pFile), label=None):
        """
            Saving the solution into the provided files. If label is specified the file will 
            be saved with that label. This can be time or frame number etc.
            Input arguments:
                (uFile, pFile): The files where the velocity and the pressure fields are saved.
                    These need to be files where dolfin can save a function. The file format .pvd
                    is probably the easiest to use.
            Optional argument:
                label: The label of the flow field. This can be time, frame number etc. It should be
                    convertible to float.
        """
        P1 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),1)  # for pressure
        P2 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),2)  # for a velocity component
        
        uSpace = dolf.FunctionSpace(self.airfoil.mesh, dolf.MixedElement([P2,P2]))
        uSave = dolf.Function(uSpace)
        dolf.assign(uSave,[self.u_x,self.u_y])
        uSave.rename('velocity', 'uSave')
        
        pSpace = dolf.FunctionSpace(self.airfoil.mesh, P1)
        pSave = dolf.Function(pSpace)
        dolf.assign(pSave, self.p)
        pSave.rename('pressure', 'pSave')
        
        if(label == None):
            uFile << uSave
            pFile << pSave
        else:
            uFile << (uSave, float(label))
            pFile << (pSave, float(label))
    
    # Making a smooth function for the Gaussian drag force
    def smoothPhi(self):
        """
            Create a smooth function space which is 1 on the inside boundary 
            and zero on the outside boundary. The function satisfies 
            del^2(Phi) = 0 inside the domain.
        """
        # Specify function spaces and functions
        P2 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),2)
        W = dolf.FunctionSpace(self.airfoil.mesh, P2)
        phi = dolf.TrialFunction(W)
        v = dolf.TestFunction(W)
        
        # Specify boundary conditions
        # Zero on outer boundary
        leftBoundary = self.airfoil.leftBoundary
        rightBoundary = self.airfoil.rightBoundary
        bottomTopBoundary = self.airfoil.bottomTopBoundary
        class OuterBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return leftBoundary(x, on_boundary) or \
                       rightBoundary(x, on_boundary) or \
                       bottomTopBoundary(x, on_boundary)
        bcPhiOuter = dolf.DirichletBC(W, 0., OuterBoundarySubDomain())
        
        # One on the inside boundary
        airfoilBoundary = self.airfoil.airfoilBoundary
        class AirfoilBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return airfoilBoundary(x, on_boundary)
        bcPhiAirfoil = dolf.DirichletBC(W, 1., AirfoilBoundarySubDomain())
        
        #Collecting the boundary conditions
        bcPhi = [bcPhiOuter, bcPhiAirfoil]
        
        # Setting the problem
        # grad(phi)*grad(v) = 0
        F = -dolf.dot(dolf.grad(v),dolf.grad(phi))*dolf.dx
        
        # Solving the problem with the appropriate boundary conditions
        phi = dolf.Function(W)
        dolf.solve(dolf.lhs(F) == dolf.rhs(F), phi, bcPhi)
        
        self.phi = phi
        
        return None
    
    def dragForceDirect(self, directionVector):
        """
            The drag force on the airfoil in the given direction.
            It is calculated directly by integrating on the airfoil boundary.
            It is assumed that directionVector has unit length.
            Input argument:
                directionVector: Unit vector. The component of the force 
                    in that direction is calculated.
        """
        a = dolf.as_vector(directionVector)
        u, p = self.u, self.p
        mu = self.mu
        n = self.airfoil.n
        dSurf = self.airfoil.dSurf
        
        # The stress tensor
        # Stress tensor T_ij = p delta(i,j) - nu*((du_i/dx_j)+(du_j/dx_i))
        T = p*dolf.Identity(2)-mu*(dolf.grad(u)+dolf.nabla_grad(u))
        
        # The drag force is t_i = T_ij*n_j on the boundary
        # In the specific direction dir
        # force = dir_i*T_ij_n_j
        elementaryForce = dolf.dot(dolf.dot(a,T),n)*dSurf(1)
        force = dolf.assemble(elementaryForce)
        
        return force
    
    def dragForceGauss(self, directionVector):
        """
            The drag force on the airfoil in the given direction.
            It is calculated using Gauss' law.
            It is assumed that directionVector has unit length.
            Input argument:
                directionVector: Unit vector. The component of the force 
                    in that direction is calculated.
        """
        # The drag force is given by a_j*T_ji*n_i
        # integrated on the inside boundary
        # Instead it is given by div(phi*a_j*T_ji)
        # integrated inside the domain
        # Using that div(a*T) = u*grad(a*u)
        a = dolf.as_vector(directionVector)
        u, p = self.u, self.p
        mu = self.mu
        rho = self.rho
        phi = self.phi
        # The stress tensor
        # Stress tensor T_ij = p delta(i,j) - nu*((du_i/dx_j)+(du_j/dx_i))
        T = p*dolf.Identity(2)-mu*(dolf.grad(u)+dolf.nabla_grad(u))
        
        elementaryForce = dolf.dot(dolf.dot(a,T),dolf.grad(phi))*dolf.dx + \
                        -rho*dolf.dot(u,dolf.grad(dolf.dot(a,u)))*phi*dolf.dx
        force = dolf.assemble(elementaryForce)
        
        return force
    
    def forceDerivative(self, directionVector):
        """
            Calculating the derivative of the drag force in the given direction 
            with respect to the parameters of the airfoil. A list of derivatives 
            is returned. It is assumed the the given direction vector has unit length.
            Input argument:
                directionVector: Unit vector. The component of the force 
                    in that direction is calculated.
        """
        a = dolf.as_vector(directionVector)
        u = self.u
        mu = self.mu
        phi = self.phi
        n = self.airfoil.n
        dSurf = self.airfoil.dSurf
        # make adjoint state
        wFull = self.adjointStateForce(a)
        (w_x,w_y,r) = dolf.split(wFull)
        w = dolf.as_vector([w_x,w_y])
        
        # The elementary change
        elementaryChange = dolf.dot(dolf.grad(dolf.dot(a,u)),dolf.grad(phi)) + \
                           -dolf.inner(dolf.grad(u),dolf.grad(w))
        elementaryChange = mu*elementaryChange
        
        derivative = []
        for i in range(self.airfoil.dim):
            V = self.airfoil.surfaceDeformation(i)
            tempDer = dolf.dot(V,n)*elementaryChange*dSurf(1)
            tempDer = dolf.assemble(tempDer)
            derivative.append(tempDer)
        
        
        return derivative

    def adjointStateForce(self, directionVector):
        """
            Adjoint state for calculating the derivative of the drag force in one direction.
        """
        a = dolf.as_vector(directionVector)
        u, p = self.u, self.p
        mu = self.mu
        rho = self.rho
        phi = self.phi
        n = self.airfoil.n
        dSurf = self.airfoil.dSurf
        # Define function space
        # (u_x, u_y, p)
        P1 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),1)  # for pressure
        P2 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),2)  # for a velocity component
        W = dolf.FunctionSpace(self.airfoil.mesh, dolf.MixedElement([P2,P2,P1]))
        
        wFull = dolf.TrialFunction(W)
        vFull = dolf.TestFunction(W)
        (w_x,w_y,r) = dolf.split(wFull)
        (v_x,v_y,q) = dolf.split(vFull)
        w = dolf.as_vector([w_x,w_y])
        v = dolf.as_vector([v_x,v_y])
        
        # Boundary conditions
        # Airfoil
        airfoilBoundary = self.airfoil.airfoilBoundary
        class AirfoilBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return airfoilBoundary(x, on_boundary)
        bcwAirfoilX = dolf.DirichletBC(W.sub(0), 0, AirfoilBoundarySubDomain())
        bcwAirfoilY = dolf.DirichletBC(W.sub(1), 0, AirfoilBoundarySubDomain())
        
        # Inflow
        leftBoundary = self.airfoil.leftBoundary
        class LeftBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return leftBoundary(x, on_boundary)
        bcwInflowX = dolf.DirichletBC(W.sub(0), 0, LeftBoundarySubDomain())
        bcwInflowY = dolf.DirichletBC(W.sub(1), 0, LeftBoundarySubDomain())
        
        # Bottom and top
        bottomTopBoundary = self.airfoil.bottomTopBoundary
        class BottomTopBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return bottomTopBoundary(x, on_boundary)
        bcwBottomTopY = dolf.DirichletBC(W.sub(1), 0, BottomTopBoundarySubDomain())
        
        bcw = [bcwAirfoilX, bcwAirfoilY, bcwInflowX, bcwInflowY, bcwBottomTopY]
        
        # Define the variational problem
        F = -rho*dolf.dot(v,dolf.grad(dolf.dot(a,u)))*phi*dolf.dx + \
            rho*dolf.dot(v,a)*dolf.dot(u,dolf.grad(phi))*dolf.dx + \
            rho*dolf.dot(v,dolf.dot(w,dolf.grad(u)))*dolf.dx + \
            -rho*dolf.dot(v,dolf.dot(u,dolf.nabla_grad(w)))*dolf.dx + \
            -mu*dolf.div(v)*dolf.dot(a,dolf.grad(phi))*dolf.dx + \
            mu*dolf.inner(dolf.grad(v),dolf.grad(w))*dolf.dx + \
            dolf.div(v)*r*dolf.dx + \
            -mu*dolf.dot(v,a)*dolf.dot(dolf.grad(phi),n)*(dSurf(3)+dSurf(4)) + \
            -mu*dolf.dot(v,dolf.grad(phi))*dolf.dot(a,n)*(dSurf(3)+dSurf(4)) + \
            mu*dolf.dot(v,n)*dolf.dot(a,dolf.grad(phi))*dSurf(4) + \
            rho*dolf.dot(v,w)*dolf.dot(u,n)*dSurf(4) + \
            q*dolf.div(w)*dolf.dx + \
            -q*dolf.dot(a,dolf.grad(phi))*dolf.dx
        
        # Solving the variational problem
        wFull = dolf.Function(W)
        dolf.solve(dolf.lhs(F) == dolf.rhs(F), wFull, bcw)
        
        return wFull
    
    def costLD(self):
        """
            The Lift-Drag force ratio of the airfoil.
        """
        L = self.dragForceGauss((0,1))
        D = self.dragForceGauss((1,0))
        
        # A repulsive term for the joukowsky airfoil to prevent overlapping
        #cCost = 0.005/(self.airfoil.joukowskyParameters[0]+0.025)
        
        return (L/D) #+ cCost

    def costLDDerivative(self):
        """
            The derivative of the Lift-Drag ratio with respect to the 
            parameters of the airfoil. A list of derivatives is returned.
        """
        L = self.dragForceGauss((0,1))
        D = self.dragForceGauss((1,0))
        derL = self.forceDerivative((0,1))
        derD = self.forceDerivative((1,0))
        
        derCost = []
        for i in range(self.airfoil.dim):
            derCost.append((derL[i]/D)-(derD[i]*L/pow(D,2)))
        
        # The derivative of the repulsive term in the case of a joukowsky airfoil
        #if(paramID == 0):
        #    derCost -= 0.005*pow(self.airfoil.joukowskyParameters[0]+0.025, -2)
        
        return derCost

    def costFunctionHessian(self, stepSize):
        """
            A now unused function to calculate the Hessian matrix of the Lift-Drag ratio.
            It is not working now.
        """
        param = self.joukowskyParameters[:]
        dim = len(joukowskyParameters)
        H = np.zeros((dim, dim))
        
        for i in range(dim):
            param[i] -= stepSize/2.
            tempJoukow = JoukowskyAirfoil(self.v_in, self.mu, self.rho, self.resolution, self.outerGeometry, param)
            costGradMinus = []
            for j in range(dim):
                costGradMinus.append(tempJoukow.costFunctionDerivative(j))
            
            param[i] += stepSize
            tempJoukow = JoukowskyAirfoil(self.v_in, self.mu, self.rho, self.resolution, self.outerGeometry, param)
            costGradPlus = []
            for j in range(dim):
                costGradPlus.append(tempJoukow.costFunctionDerivative(j))
            
            for j in range(dim):
                secondDer = (costGradPlus[j] - costGradMinus[j])/stepSize
                H[i][j] = secondDer
            param[:] = self.joukowskyParameters[:]
        
        # Symmetrise the Hessian
        H = 0.5*(H + H.T)
        
        return H
    
    def findEigenvalues(self, nEigenvalues=60):
        """
            Search for eigenvalues and eigenflows. As a default the first 60 eigenvalues are searched.
            This is not surely enough to find the most unstable mode. The following variables are created:
                eigenvalues: List of (sReal, sImag) eigenvalue pairs.
                eigenflows: List of (wReal, wImag) eigenflow pairs.
            Optional argument:
                nEigenvalues: The number of eigenvalues to be found. Default is 60.
        """
        
        rho = self.rho
        mu = self.mu
        u = self.u
        
        
        # Define function space
        P1 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),1)  # for pressure
        P2 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),2)  # for a velocity component
        W = dolf.FunctionSpace(self.airfoil.mesh, dolf.MixedElement([P2,P2,P1]))
        
        # Trial and test functions
        wFull = dolf.TrialFunction(W)
        vFull = dolf.TestFunction(W)
        (w_x,w_y,r) = dolf.split(wFull)
        (v_x,v_y,q) = dolf.split(vFull)
        w = dolf.as_vector([w_x,w_y])
        v = dolf.as_vector([v_x,v_y])
        
        # Boundary conditions for the pertubed flow
        # Inflow velocity
        leftBoundary = self.airfoil.leftBoundary
        class LeftBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return leftBoundary(x, on_boundary)
        bcwInflowX = dolf.DirichletBC(W.sub(0), 0, LeftBoundarySubDomain())
        bcwInflowY = dolf.DirichletBC(W.sub(1), 0, LeftBoundarySubDomain())
        
        # Zero y velocity at the bottom and top
        bottomTopBoundary = self.airfoil.bottomTopBoundary
        class BottomTopBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return bottomTopBoundary(x, on_boundary)
        bcwBottomTopY = dolf.DirichletBC(W.sub(1), 0, BottomTopBoundarySubDomain())
        
        # No slip at the airfoil
        airfoilBoundary = self.airfoil.airfoilBoundary
        class AirfoilBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return airfoilBoundary(x, on_boundary)
        bcwAirfoilX = dolf.DirichletBC(W.sub(0), 0, AirfoilBoundarySubDomain())
        bcwAirfoilY = dolf.DirichletBC(W.sub(1), 0, AirfoilBoundarySubDomain())
        
        bcw = [bcwInflowX, bcwInflowY, bcwBottomTopY, bcwAirfoilX, bcwAirfoilY]
        
        # Setting the problem
        # The continuous problem
        # Left hand side
        A = rho*dolf.dot(v, dolf.dot(u, dolf.nabla_grad(w)))*dolf.dx + \
            rho*dolf.dot(v, dolf.dot(w, dolf.nabla_grad(u)))*dolf.dx + \
            -dolf.div(v)*r*dolf.dx + \
            mu*dolf.inner(dolf.grad(v), dolf.grad(w))*dolf.dx + \
            q*dolf.div(w)*dolf.dx
        
        # In matrix form
        AM = dolf.PETScMatrix()
        dolf.assemble(A, tensor=AM)
        
        # Right hand side
        B = -rho*dolf.dot(v, w)*dolf.dx
        
        # In matrix form
        BM = dolf.PETScMatrix()
        dolf.assemble(B, tensor=BM)
        
        # Apply the boundary conditions
        for bc in bcw:
            bc.apply(AM)
            bc.zero(BM)
        
        # Initialise the eigenvalue solver
        eigensolver = dolf.SLEPcEigenSolver(AM,BM);
        
        # Set the eigensolver parameters
        eigensolver.parameters['spectrum']  = 'smallest magnitude';
        eps     =eigensolver.eps();
        st      = eps.getST();
        st.setType('sinvert');
        ksp     = st.getKSP();
        ksp.setType('preonly');
        pc      = ksp.getPC();
        pc.setType('lu');
        pc.setFactorSolverPackage('mumps');
        
        eigensolver.solve(nEigenvalues);
        
        self.eigenvalues = []
        self.eigenflows = []
        
        for i in range(eigensolver.get_number_converged()):
            sReal, sImag, wRealFullVector, wImagFullVector = eigensolver.get_eigenpair(i)
            self.eigenvalues.append([sReal, sImag])
            
            wRealFull = dolf.Function(W)
            wImagFull = dolf.Function(W)
            
            wRealFull.vector()[:] = wRealFullVector[:]
            wImagFull.vector()[:] = wImagFullVector[:]
            
            self.eigenflows.append([wRealFull, wImagFull])
        
        print("%i eigenvaues converged"%(eigensolver.get_number_converged()))
    
    def getMaxEigenvalue(self, nonZeroFreq=False, nEigenvalues=60):
        """
            Find the most unstable eigenvalue i.e. the one with the largest real part. 
            If nonZeroFreq is True the most unstable eigenmode with positive imaginary 
            part is searched.
            Optional arguments:
                nonZeroFreq: If it is True only those eigenvalues are considered which 
                    have positive imaginary part. Default is False.
                nEigenvalues: The number of eigenvalues to be searched. Default is 60.
            Returned:
                (maxSReal,maxSImag), maxSIndex: The real and imaginary part of the found
                    eigenvalue and its index in the list of eigenvalues.
        """
        freqTol = 1E-14
        
        self.findEigenvalues(nEigenvalues)
        
        maxSReal = float('-inf')
        maxSImag = 0.
        maxSIndex = -1
        for i in range(len(self.eigenvalues)):
            tempSReal = self.eigenvalues[i][0]
            tempSImag = self.eigenvalues[i][1]
            
            if(nonZeroFreq):
                if(abs(tempSImag) > freqTol and tempSReal > maxSReal):
                    maxSReal = tempSReal
                    maxSImag = tempSImag
                    maxSIndex = i
            else:
                if(tempSReal > maxSReal):
                    maxSReal = tempSReal
                    maxSImag = tempSImag
                    maxSIndex = i
            
        self.maxSIndex = maxSIndex
        
        return (maxSReal,maxSImag), maxSIndex
    
    def eigenvalueDerivative(self, eigenIndex=None):
        """
            Find the derivative of an eigenvalue with respect to shape parameters.
            A list of the derivative of the real and imaginary parts are returned. 
            If eigenIndex is specified than the derivative of the eigenvalue of 
            that index is calculated. Otherwise the derivative of the most unstable 
            mode is calculated.
            Optional argument:
                eigenIndex: The index of the eigenvalue. Default is None when the 
                    deivative of the most unstable eigenvalue is calculated.
            Returned:
                sRealDer, sImagDer: The list of the derivatives of the real and 
                    imaginary parts with respect to the shape parameters.
        """
        if(eigenIndex==None):
            eigenIndex = self.maxSIndex
        
        u = self.u
        mu = self.mu
        
        n = self.airfoil.n
        dSurf = self.airfoil.dSurf
        
        # Find the adjoint states
        # Adjoint state of perturbance (complex conjugated)
        wAdjRealFull, wAdjImagFull = self.adjointStatePerturbance(eigenIndex)
        (wAdjReal_x,wAdjReal_y,rAdjReal) = dolf.split(wAdjRealFull)
        (wAdjImag_x,wAdjImag_y,rAdjImag) = dolf.split(wAdjImagFull)
        wAdjReal = dolf.as_vector([wAdjReal_x,wAdjReal_y])
        wAdjImag = dolf.as_vector([wAdjImag_x,wAdjImag_y])
        
        # Normalise the eigenflow
        wRealFull, wImagFull = self.normalisedEigenflow(eigenIndex, (wAdjRealFull,wAdjImagFull))
        (wReal_x, wReal_y, rReal) = dolf.split(wRealFull)
        wReal = dolf.as_vector([wReal_x,wReal_y])
        (wImag_x, wImag_y, rImag) = dolf.split(wImagFull)
        wImag = dolf.as_vector([wImag_x,wImag_y])
        
        # Adjoint state of base flow (complex conjugated)
        uAdjFull = self.adjointStateBase([wRealFull,wImagFull], (wAdjRealFull,wAdjImagFull))
        (uAdjReal_x,uAdjReal_y,pAdjReal, uAdjImag_x,uAdjImag_y,pAdjImag) = dolf.split(uAdjFull)
        uAdjReal = dolf.as_vector([uAdjReal_x,uAdjReal_y])
        uAdjImag = dolf.as_vector([uAdjImag_x,uAdjImag_y])
        
        # The real part
        sRealChange = dolf.inner(dolf.grad(u), dolf.grad(uAdjReal)) + \
                      dolf.inner(dolf.grad(wReal), dolf.grad(wAdjReal)) + \
                      -dolf.inner(dolf.grad(wImag), dolf.grad(wAdjImag))
        sRealChange = -mu*sRealChange
        
        # The imaginary part
        sImagChange = dolf.inner(dolf.grad(u), dolf.grad(uAdjImag)) + \
                      dolf.inner(dolf.grad(wReal), dolf.grad(wAdjImag)) + \
                      dolf.inner(dolf.grad(wImag), dolf.grad(wAdjReal))
        sImagChange = -mu*sImagChange
        
        sRealDer = []
        sImagDer = []
        for i in range(self.airfoil.dim):
            V = self.airfoil.surfaceDeformation(i)
            
            tempRealDer = dolf.dot(V,n)*sRealChange*dSurf(1)
            tempRealDer = dolf.assemble(tempRealDer)
            sRealDer.append(tempRealDer)
            
            tempImagDer = dolf.dot(V,n)*sImagChange*dSurf(1)
            tempImagDer = dolf.assemble(tempImagDer)
            sImagDer.append(tempImagDer)
        
        
        return sRealDer, sImagDer
    
    def adjointStatePerturbance(self, eigenIndex):
        """
            The adjoint perturbance state for calculating the derivative of an eigenvalue.
        """
        sReal, sImag = self.eigenvalues[eigenIndex]
        
        rho = self.rho
        mu = self.mu
        u = self.u
        
        n = self.airfoil.n
        dSurf = self.airfoil.dSurf
        
        # Define function space
        P1 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),1)  # for pressure
        P2 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),2)  # for a velocity component
        W = dolf.FunctionSpace(self.airfoil.mesh, dolf.MixedElement([P2,P2,P1]))
        
        wAdjFull = dolf.TrialFunction(W)
        vFull = dolf.TestFunction(W)
        (wAdj_x,wAdj_y,rAdj) = dolf.split(wAdjFull)
        (v_x,v_y,q) = dolf.split(vFull)
        wAdj = dolf.as_vector([wAdj_x,wAdj_y])
        v = dolf.as_vector([v_x,v_y])
        
        # Boundary conditions
        # Inflow
        leftBoundary = self.airfoil.leftBoundary
        class LeftBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return leftBoundary(x, on_boundary)
        bcwAdjInflowX = dolf.DirichletBC(W.sub(0), 0, LeftBoundarySubDomain())
        bcwAdjInflowY = dolf.DirichletBC(W.sub(1), 0, LeftBoundarySubDomain())
        
        # Zero y velocity at the bottom and top
        bottomTopBoundary = self.airfoil.bottomTopBoundary
        class BottomTopBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return bottomTopBoundary(x, on_boundary)
        bcwAdjBottomTopY = dolf.DirichletBC(W.sub(1), 0, BottomTopBoundarySubDomain())
        
        # No slip at the airfoil
        airfoilBoundary = self.airfoil.airfoilBoundary
        class AirfoilBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return airfoilBoundary(x, on_boundary)
        bcwAdjAirfoilX = dolf.DirichletBC(W.sub(0), 0, AirfoilBoundarySubDomain())
        bcwAdjAirfoilY = dolf.DirichletBC(W.sub(1), 0, AirfoilBoundarySubDomain())
        
        bcwAdj = [bcwAdjInflowX, bcwAdjInflowY, bcwAdjBottomTopY, bcwAdjAirfoilX, bcwAdjAirfoilY]
        
        # Setting the variational problem
        # The continuous problem
        # Left hand side
        # The continuous problem
        A = rho*dolf.dot(v, dolf.dot(wAdj,dolf.grad(u)))*dolf.dx + \
            -rho*dolf.dot(v, dolf.dot(u,dolf.nabla_grad(wAdj)))*dolf.dx + \
            mu*dolf.inner(dolf.grad(v),dolf.grad(wAdj))*dolf.dx + \
            dolf.div(v)*rAdj*dolf.dx + \
            rho*dolf.dot(v,wAdj)*dolf.dot(u,n)*dSurf(4) + \
            q*dolf.div(wAdj)*dolf.dx
        
        # Matrix form
        AM = dolf.PETScMatrix()
        dolf.assemble(A, tensor=AM)
        
        # The right hand side
        # The continuous problem
        B = -rho*dolf.dot(v,wAdj)*dolf.dx
        
        # Matrix form
        BM = dolf.PETScMatrix()
        dolf.assemble(B, tensor=BM)
        
        # Apply the boundary conditions
        for bc in bcwAdj:
            bc.apply(AM)
            bc.zero(BM)
        
        # Initialise the eigenvalue solver
        eigensolver = dolf.SLEPcEigenSolver(AM,BM);
        
        # Set the eigensolver parameters
        eigensolver.parameters['spectrum']  = 'smallest magnitude';
        eps     =eigensolver.eps();
        st      = eps.getST();
        st.setType('sinvert');
        ksp     = st.getKSP();
        ksp.setType('preonly');
        pc      = ksp.getPC();
        pc.setType('lu');
        pc.setFactorSolverPackage('mumps');
        
        # The number of eigenvalues to find
        nEigenvalues = max(60, eigenIndex+5)
        
        eigensolver.solve(nEigenvalues);
        
        def distSq(eigenReal, eigenImag):
            return pow(eigenReal-sReal,2)+pow(eigenImag-sImag,2)
        
        sAdjReal, sAdjImag, wAdjRealFullVector, wAdjImagFullVector = eigensolver.get_eigenpair(0)
        minDiffSq = distSq(sAdjReal, sAdjImag)
        
        for i in range(eigensolver.get_number_converged()):
            sTempReal, sTempImag, wTempRealFullVector, wTempImagFullVector = eigensolver.get_eigenpair(i)
            
            if(distSq(sTempReal, sTempImag) < minDiffSq):
                minDiffSq = distSq(sTempReal, sTempImag)
                
                sAdjReal = sTempReal
                sAdjImag = sTempImag
                wAdjRealFullVector = wTempRealFullVector
                wAdjImagFullVector = wTempImagFullVector
            
        wAdjRealFull = dolf.Function(W)
        wAdjImagFull = dolf.Function(W)
        
        wAdjRealFull.vector()[:] = wAdjRealFullVector[:]
        wAdjImagFull.vector()[:] = wAdjImagFullVector[:]
        
        return wAdjRealFull, wAdjImagFull
    
    def normalisedEigenflow(self, eigenIndex, (wAdjRealFull,wAdjImagFull)):
        """
            Normalising the eigenflow. It is needed for the calculation of the 
            derivative of the eigenvalue.
        """
        rho = self.rho
        
        # The perturbance
        wOrigRealFull, wOrigImagFull = self.eigenflows[eigenIndex]
        (wOrigReal_x, wOrigReal_y, rOrigReal) = dolf.split(wOrigRealFull)
        wOrigReal = dolf.as_vector([wOrigReal_x,wOrigReal_y])
        (wOrigImag_x, wOrigImag_y, rOrigImag) = dolf.split(wOrigImagFull)
        wOrigImag = dolf.as_vector([wOrigImag_x,wOrigImag_y])
        
        # The adjoint state
        (wAdjReal_x,wAdjReal_y,rAdjReal) = dolf.split(wAdjRealFull)
        (wAdjImag_x,wAdjImag_y,rAdjImag) = dolf.split(wAdjImagFull)
        wAdjReal = dolf.as_vector([wAdjReal_x,wAdjReal_y])
        wAdjImag = dolf.as_vector([wAdjImag_x,wAdjImag_y])
        
        # The integral prod = rho*wAdj*wOrig*dx
        # The real part
        prodReal = rho*dolf.dot(wAdjReal,wOrigReal)*dolf.dx + \
                   -rho*dolf.dot(wAdjImag, wOrigImag)*dolf.dx
        prodReal = dolf.assemble(prodReal)
        
        # The imaginary part
        prodImag = rho*dolf.dot(wAdjImag, wOrigReal)*dolf.dx + \
                   rho*dolf.dot(wAdjReal, wOrigImag)*dolf.dx
        prodImag = dolf.assemble(prodImag)
        
        # The multiplicative factor for w is -1/prod
        alphaReal = -prodReal / (pow(prodReal,2)+pow(prodImag,2))
        alphaImag =  prodImag / (pow(prodReal,2)+pow(prodImag,2))
        
        # Define function space
        P1 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),1)  # for pressure
        P2 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),2)  # for a velocity component
        W = dolf.FunctionSpace(self.airfoil.mesh, dolf.MixedElement([P2,P2,P1]))
        
        wRealFull = dolf.Function(W)
        wImagFull = dolf.Function(W)
        
        wRealFull = dolf.project(alphaReal*wOrigRealFull - alphaImag*wOrigImagFull, W)
        wImagFull = dolf.project(alphaReal*wOrigImagFull + alphaImag*wOrigRealFull, W)
        
        return wRealFull, wImagFull
    
    def adjointStateBase(self, (wRealFull,wImagFull), (wAdjRealFull,wAdjImagFull)):
        """
            The adjoint base state required for the derivative of the eigenvalue.
        """
        rho = self.rho
        mu = self.mu
        u = self.u
        
        n = self.airfoil.n
        dSurf = self.airfoil.dSurf
        
        (wReal_x, wReal_y, rReal) = dolf.split(wRealFull)
        wReal = dolf.as_vector([wReal_x,wReal_y])
        (wImag_x, wImag_y, rImag) = dolf.split(wImagFull)
        wImag = dolf.as_vector([wImag_x,wImag_y])
        
        (wAdjReal_x,wAdjReal_y,rAdjReal) = dolf.split(wAdjRealFull)
        (wAdjImag_x,wAdjImag_y,rAdjImag) = dolf.split(wAdjImagFull)
        wAdjReal = dolf.as_vector([wAdjReal_x,wAdjReal_y])
        wAdjImag = dolf.as_vector([wAdjImag_x,wAdjImag_y])
        
        # Define function space
        P1 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),1)  # for pressure
        P2 = dolf.FiniteElement("CG",self.airfoil.mesh.ufl_cell(),2)  # for a velocity component
        W = dolf.FunctionSpace(self.airfoil.mesh, dolf.MixedElement([P2,P2,P1,P2,P2,P1]))
        
        uAdjFull = dolf.TrialFunction(W)
        vFull = dolf.TestFunction(W)
        (uAdjReal_x,uAdjReal_y,pAdjReal, uAdjImag_x,uAdjImag_y,pAdjImag) = dolf.split(uAdjFull)
        (vReal_x,vReal_y,qReal, vImag_x,vImag_y,qImag) = dolf.split(vFull)
        uAdjReal = dolf.as_vector([uAdjReal_x, uAdjReal_y])
        uAdjImag = dolf.as_vector([uAdjImag_x, uAdjImag_y])
        vReal = dolf.as_vector([vReal_x, vReal_y])
        vImag = dolf.as_vector([vImag_x, vImag_y])
        
        # Boundary conditions
        # Inflow
        leftBoundary = self.airfoil.leftBoundary
        class LeftBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return leftBoundary(x, on_boundary)
        bcuAdjRealInflowX = dolf.DirichletBC(W.sub(0), 0, LeftBoundarySubDomain())
        bcuAdjRealInflowY = dolf.DirichletBC(W.sub(1), 0, LeftBoundarySubDomain())
        bcuAdjImagInflowX = dolf.DirichletBC(W.sub(3), 0, LeftBoundarySubDomain())
        bcuAdjImagInflowY = dolf.DirichletBC(W.sub(4), 0, LeftBoundarySubDomain())
        
        # Zero y velocity at the bottom and top
        bottomTopBoundary = self.airfoil.bottomTopBoundary
        class BottomTopBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return bottomTopBoundary(x, on_boundary)
        bcuAdjRealBottomTopY = dolf.DirichletBC(W.sub(1), 0, BottomTopBoundarySubDomain())
        bcuAdjImagBottomTopY = dolf.DirichletBC(W.sub(4), 0, BottomTopBoundarySubDomain())
        
        # No slip at the airfoil
        airfoilBoundary = self.airfoil.airfoilBoundary
        class AirfoilBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return airfoilBoundary(x, on_boundary)
        bcuAdjRealAirfoilX = dolf.DirichletBC(W.sub(0), 0, AirfoilBoundarySubDomain())
        bcuAdjRealAirfoilY = dolf.DirichletBC(W.sub(1), 0, AirfoilBoundarySubDomain())
        bcuAdjImagAirfoilX = dolf.DirichletBC(W.sub(3), 0, AirfoilBoundarySubDomain())
        bcuAdjImagAirfoilY = dolf.DirichletBC(W.sub(4), 0, AirfoilBoundarySubDomain())
        
        bcuAdj = [bcuAdjRealInflowX, bcuAdjRealInflowY, bcuAdjImagInflowX, bcuAdjImagInflowY]
        bcuAdj += [bcuAdjRealBottomTopY, bcuAdjImagBottomTopY]
        bcuAdj += [bcuAdjRealAirfoilX, bcuAdjRealAirfoilY, bcuAdjImagAirfoilX, bcuAdjImagAirfoilY]
        
        # Variational form for real part
        FReal = -rho*dolf.dot(vReal, dolf.dot(u,dolf.nabla_grad(uAdjReal)))*dolf.dx + \
                rho*dolf.dot(vReal, dolf.dot(uAdjReal,dolf.grad(u)))*dolf.dx + \
                mu*dolf.inner(dolf.grad(vReal),dolf.grad(uAdjReal))*dolf.dx + \
                dolf.div(vReal)*pAdjReal*dolf.dx + \
                rho*dolf.dot(vReal,uAdjReal)*dolf.dot(u,n)*dSurf(4) + \
                rho*dolf.dot(vReal,wAdjReal)*dolf.dot(wReal,n)*dSurf(4) + \
                -rho*dolf.dot(vReal,wAdjImag)*dolf.dot(wImag,n)*dSurf(4) + \
                rho*dolf.dot(vReal, dolf.dot(wAdjReal,dolf.grad(wReal)))*dolf.dx + \
                -rho*dolf.dot(vReal, dolf.dot(wAdjImag,dolf.grad(wImag)))*dolf.dx + \
                -rho*dolf.dot(vReal, dolf.dot(wReal,dolf.nabla_grad(wAdjReal)))*dolf.dx + \
                rho*dolf.dot(vReal, dolf.dot(wImag, dolf.nabla_grad(wAdjImag)))*dolf.dx + \
                qReal*dolf.div(uAdjReal)*dolf.dx
        
        # Variational form for imaginary part
        FImag = -rho*dolf.dot(vImag, dolf.dot(u,dolf.nabla_grad(uAdjImag)))*dolf.dx + \
                rho*dolf.dot(vImag, dolf.dot(uAdjImag,dolf.grad(u)))*dolf.dx + \
                mu*dolf.inner(dolf.grad(vImag),dolf.grad(uAdjImag))*dolf.dx + \
                dolf.div(vImag)*pAdjImag*dolf.dx + \
                rho*dolf.dot(vImag,uAdjImag)*dolf.dot(u,n)*dSurf(4) + \
                rho*dolf.dot(vImag,wAdjImag)*dolf.dot(wReal,n)*dSurf(4) + \
                rho*dolf.dot(vImag,wAdjReal)*dolf.dot(wImag,n)*dSurf(4) + \
                rho*dolf.dot(vImag, dolf.dot(wAdjImag,dolf.grad(wReal)))*dolf.dx + \
                rho*dolf.dot(vImag, dolf.dot(wAdjReal,dolf.grad(wImag)))*dolf.dx + \
                -rho*dolf.dot(vImag, dolf.dot(wImag,dolf.nabla_grad(wAdjReal)))*dolf.dx + \
                -rho*dolf.dot(vImag, dolf.dot(wReal,dolf.nabla_grad(wAdjImag)))*dolf.dx + \
                qImag*dolf.div(uAdjImag)*dolf.dx
                
        F = FReal+FImag
        
        # Solving the variational problem
        uAdjFull = dolf.Function(W)
        dolf.solve(dolf.lhs(F) == dolf.rhs(F), uAdjFull, bcuAdj)
        
        return uAdjFull
    
    
