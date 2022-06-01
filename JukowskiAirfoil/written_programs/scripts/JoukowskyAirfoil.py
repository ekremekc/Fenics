import pygmsh as pg
import math
import cmath
import os
import dolfin as dolf
import numpy as np
import sympy as sp

class JoukowskyAirfoil:
    """
        A Joukowsky-like airfoil. (dim = 3)
        Created fields:
            mesh: The mesh of the domain.
            dim: The number of parameters of the airfoil.
            area: The area of the airfoil.
            airfoilPoints: The list of (x,y) coordinates of the vertices of the airfoil.
            deltax: The width of the airfoil.
            deltay: The thickness of the airfoil.
            dSurf: The object one can use to integrate on a specific boundary.
            n: The surface normal.
        Callable functions:
            leftBoundary(): The inflow boundary.
            rightBoundary(): The outflow boundary.
            bottomTopBoundary(): The wall boundary.
            airfoilBoundary(): The boundary at the airfoil.
            surfaceDeformation(): The deformation vector space of the surface.
            areaDerivative():  A list of the derivatives of area with respect 
                to the different shape parameters.
    """
    
    def __init__(self, resolution, outerGeometry, joukowskyParameters):
        """
            Initialising arguments:
                resolution: The number of points on the airfoil.
                outerGeometry: The sizes of the outer bounding box.
                joukowskyParameters: The parameters of the joukowsky airfoil.
        """
        # Setting the variables
        self.resolution = resolution
        self.outerGeometry = outerGeometry
        self.joukowskyParameters = joukowskyParameters
        # The number of parameters
        self.dim = 3
        
        meshName = "mesh/joukowsky_mesh_"+str(resolution)
        
        # Creating the mesh
        self.createJoukowskyMesh(meshName)
        
        # The area of the airfoil
        self.getArea()
        
        # Surface normal
        self.n = dolf.FacetNormal(self.mesh)
        
        # Marking the boundaries
        self.markedBoundaries()
    
    # Defining boundaries
    def leftBoundary(self, x, on_boundary):
        return on_boundary and dolf.near(x[0], -self.outerGeometry["xFrontLength"])
    
    def rightBoundary(self, x, on_boundary):
        return on_boundary and dolf.near(x[0], self.outerGeometry["xBackLength"])
    
    def bottomTopBoundary(self, x, on_boundary):
        return on_boundary and (dolf.near(x[1], -self.outerGeometry["yHalfLength"]) \
                                or dolf.near(x[1], self.outerGeometry["yHalfLength"]))
    
    def airfoilBoundary(self, x, on_boundary):
        return on_boundary and (2*x[0] >= -self.outerGeometry["xFrontLength"] and \
                                2*x[0] <= self.outerGeometry["xBackLength"] and \
                                2*x[1] >= -self.outerGeometry["yHalfLength"] and \
                                2*x[1] <= self.outerGeometry["yHalfLength"])
    
    def createJoukowskyMesh(self, meshName, saveMesh=False):
        """
            Make a mesh around a Joukowsky airfoil. The mesh is also saved 
            as a .xml file if saveMesh is True. It is important not to create two meshes at 
            the same time with the same name.
            Input argument:
                meshName: The name of the mesh. If saveMesh is True the mesh is saved as meshName.xml.
            Optional argument:
                saveMesh: If it is True the mesh is saved as an .xml file.
            Created fields:
                deltax: The width of the airfoil.
                deltay: The thickness of the airfoil.
                airfoilPoints: The list of (x,y) coordinates of the vertices of the airfoil.
                mesh: The mesh of the domain.
        """
        geom = pg.Geometry()
        outerCircumference = (self.outerGeometry["xFrontLength"] + self.outerGeometry["xBackLength"])*2.0 + \
                              self.outerGeometry["yHalfLength"]*4.0
        
        # Table for airfoil ponts
        self.globalJoukowskyTable = []
        
        # Angles of the airfoil
        airfoilAngles = np.linspace(0., 2.0*math.pi, self.resolution, endpoint=False)
        
        xMin = 0.
        xMax = 0.
        yMin = 0.
        yMax = 0.
        
        # Points on the airfoil
        airfoilPoints = []
        self.airfoilPoints = []
        for angle in airfoilAngles:
            (xAirfoil,yAirfoil) = self.joukowskyTransform(angle)
            self.airfoilPoints.append((xAirfoil,yAirfoil))
            airfoilPoints.append((xAirfoil,yAirfoil,0.0))
            
            # Update global lookup table
            self.globalJoukowskyTable.append(((xAirfoil,yAirfoil), angle))
            
            # Update the extremal coordinate values
            if(xAirfoil < xMin):
                xMin = xAirfoil
            if(xAirfoil > xMax):
                xMax = xAirfoil
            if(yAirfoil < yMin):
                yMin = yAirfoil
            if(yAirfoil > yMax):
                yMax = yAirfoil
        
        # Thickness and length
        self.deltax = xMax-xMin
        self.deltay = yMax-yMin
        
        # The airfoil surface.
        # The characteristic length sould be larger than the actual distance between the points.
        airfoil = geom.add_polygon(airfoilPoints, outerCircumference, make_surface = False)
        
        bBoxPoints = airfoilPoints
        bBox = airfoil
        # Bounding boxes:
        # Small bounding box
        bBoxPoints = [[xMin-self.deltax/2., yMax+self.deltay/2., 0.0], \
                      [xMin-self.deltax/2., yMin-self.deltay/2., 0.0], \
                      [xMax+self.deltax/2., yMin-self.deltay/2., 0.0], \
                      [xMax+self.deltax/2., yMax+self.deltay/2., 0.0]]
        bBox = geom.add_polygon(bBoxPoints, 2.*(self.deltax+self.deltay)/self.resolution, holes=[bBox])
        # Middle bounding box
        bBoxPoints = [[-0.5*(self.outerGeometry["xFrontLength"]-xMin+self.deltax/2.), \
                        0.5*(self.outerGeometry["yHalfLength"]+yMax+self.deltay/2.), 0.0], \
                    [-0.5*(self.outerGeometry["xFrontLength"]-xMin+self.deltax/2.), \
                        -0.5*(self.outerGeometry["yHalfLength"]-yMin+self.deltay/2.), 0.0], \
                    [ 0.5*(self.outerGeometry["xBackLength"]+xMax+self.deltax/2.), \
                        -0.5*(self.outerGeometry["yHalfLength"]-yMin+self.deltay/2.), 0.0], \
                    [ 0.5*(self.outerGeometry["xBackLength"]+xMax+self.deltax/2.), \
                        0.5*(self.outerGeometry["yHalfLength"]+yMax+self.deltay/2.), 0.0]]
        bBox = geom.add_polygon(bBoxPoints, 0.5*outerCircumference/self.resolution, holes=[bBox])
        # Large bounding box
        bBoxPoints = [[-self.outerGeometry["xFrontLength"],  self.outerGeometry["yHalfLength"], 0.0], \
                      [-self.outerGeometry["xFrontLength"], -self.outerGeometry["yHalfLength"], 0.0], \
                      [ self.outerGeometry["xBackLength"],  -self.outerGeometry["yHalfLength"], 0.0], \
                      [ self.outerGeometry["xBackLength"],   self.outerGeometry["yHalfLength"], 0.0]]
        bBox = geom.add_polygon(bBoxPoints, outerCircumference/self.resolution, holes=[bBox])
        
        # Exporting the geometry
        gmshOutput = open(meshName+".geo", 'w')
        gmshOutput.write(geom.get_code())
        gmshOutput.close()
    
        # Creating mesh
        os.system("./gmsh %s -2 -o %s -v 3"%(meshName+".geo", meshName+".msh"))
        
        # Converting the mesh to .xml file
        os.system('dolfin-convert %s %s'%(meshName+".msh", meshName+".xml"))
        
        # Reading the mesh
        mesh = dolf.Mesh(meshName+".xml")
        
        # Delete unnecessary files
        os.system("rm %s"%(meshName+".geo"))
        os.system("rm %s"%(meshName+".msh"))
        if(not(saveMesh)):
            os.system("rm %s"%(meshName+".xml"))
        
        self.mesh = mesh
        
        return None

    def joukowskyTransform(self, theta):
        """
            Get the airfoil point's coordinates from the angle theta.
        """
        # Set this parameter to avoid the sharp trailing edge
        b = -0.05
        cReal = self.joukowskyParameters[0]
        cImag = self.joukowskyParameters[1]
        c = complex(cReal, cImag)
        alpha = self.joukowskyParameters[2]
        
        z = b + (1.+c)*complex(math.cos(theta),math.sin(theta)) - c
        zeta = z + 1./z
        tf = zeta*complex(math.cos(alpha),math.sin(alpha))
        tfX = tf.real
        tfY = tf.imag
        
        return (tfX,tfY)
    
    def inverseJoukowskyTransform(self, coords):
        """
            Get the theta angle from the coordinates of the airfoil point.
        """
        def distSq((x,y)):
            return(pow(x-coords[0], 2) + pow(y-coords[1], 2))
        
        minDistSq = distSq(self.globalJoukowskyTable[0][0])
        angle = self.globalJoukowskyTable[0][1]
        for i in range(1, len(self.globalJoukowskyTable)):
            tempDistSq = distSq(self.globalJoukowskyTable[i][0])
            if(tempDistSq < minDistSq):
                minDistSq = tempDistSq
                angle = self.globalJoukowskyTable[i][1]
        
        return angle

    def airfoilShapeDerivative(self, theta, paramID):
        """
            The surface deformation of a point of the airfoil of 
            angle theta with respect to the change of the paramID^th
            parameter.
        """
        diffParam = 'p'+str(paramID)
        # Set this parameter to avoid the sharp trailing edge
        b = -0.05
        (cRealSp, cImagSp, alphaSp) = sp.symbols('p0 p1 p2')
        
        cSp = cRealSp + sp.I*cImagSp
        
        zSp = b + (1.+cSp)*sp.exp(sp.I*theta) - cSp
        zetaSp = zSp + 1./zSp
        tfSp = zetaSp*sp.exp(sp.I*alphaSp)
        
        
        tfDerSp = tfSp.diff(diffParam)
        tfDer = tfDerSp.subs('p0', self.joukowskyParameters[0])
        tfDer = tfDer.subs('p1', self.joukowskyParameters[1])
        tfDer = tfDer.subs('p2', self.joukowskyParameters[2])
        tfDer = tfDer.evalf()
        tfDer = complex(tfDer)
        
        return (tfDer.real, tfDer.imag)

    # Marking the boundaries
    def markedBoundaries(self):
        """
            Marks the different boundaries of the airfoil.
            1: Airfoil boundary
            2: Left boundary
            3: Bottom and top boundary
            4: Right boundary
            Created fields:
                dSurf: The object one ca use to integrate on a specific boundary.
        """
        boundaries = dolf.FacetFunction("size_t", self.mesh)
        boundaries.set_all(0)
        
        # Airfoil boundary
        airfoilBoundary = self.airfoilBoundary
        class AirfoilBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return airfoilBoundary(x, on_boundary)
        airfoilB = AirfoilBoundarySubDomain()
        airfoilB.mark(boundaries, 1)
        
        # Left boundary
        leftBoundary = self.leftBoundary
        class LeftBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return leftBoundary(x, on_boundary)
        LeftB = LeftBoundarySubDomain()
        LeftB.mark(boundaries, 2)
        
        # Bottom and top boundary
        bottomTopBoundary = self.bottomTopBoundary
        class BottomTopBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return bottomTopBoundary(x, on_boundary)
        bottomTopB = BottomTopBoundarySubDomain()
        bottomTopB.mark(boundaries, 3)
        
        # Right boundary
        rightBoundary = self.rightBoundary
        class RightBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return rightBoundary(x, on_boundary)
        rightB = RightBoundarySubDomain()
        rightB.mark(boundaries, 4)
        
        self.dSurf = dolf.Measure("ds",subdomain_data=boundaries)
        
        return None
    
    def surfaceDeformation(self, paramID):
        """
            The vector space of deformation of surface of the airfoil with respect to 
            the paramID^th parameter. It is needed for the calculation of the 
            derivative of the cost function using the adjoint method.
        """
        P1 = dolf.FiniteElement("CG",self.mesh.ufl_cell(),1)
        W = dolf.FunctionSpace(self.mesh, P1)
        Vx = dolf.Function(W)
        Vy = dolf.Function(W)
        
        # Connecting the vertices to the DoF
        vertexToDoF = dolf.vertex_to_dof_map(W)
        
        # Function which is True on the airfoil
        # Need to use FacetFunction here
        fFunction = dolf.FacetFunctionBool(self.mesh, False)
        
        airfoilBoundary = self.airfoilBoundary
        class AirfoilBoundarySubDomain(dolf.SubDomain):
            def inside(self, x, on_boundary):
                return airfoilBoundary(x, on_boundary)
               
        airfoilB = AirfoilBoundarySubDomain()
        airfoilB.mark(fFunction, True)
        
        for f in dolf.facets(self.mesh):
            if fFunction[f]:
                for v in dolf.vertices(f):
                    theta = self.inverseJoukowskyTransform((v.x(0), v.x(1)))
                    (dX, dY) = self.airfoilShapeDerivative(theta, paramID)
                    
                    # Setting the values of the functions
                    dofIndex = vertexToDoF[v.index()]
                    Vx.vector()[dofIndex] = dX
                    Vy.vector()[dofIndex] = dY
                
        # Creating a vector space
        V = dolf.as_vector([Vx,Vy])
        
        return V
    
    def getArea(self):
        """
            Getting the area of the airfoil.
            Created fields:
                area: The area of the airfoil.
        """
        boxArea = (self.outerGeometry["xFrontLength"]+self.outerGeometry["xBackLength"])*2.*self.outerGeometry["yHalfLength"]
        
        P2 = dolf.FiniteElement("CG",self.mesh.ufl_cell(),2)  # for a velocity component
        W = dolf.FunctionSpace(self.mesh, P2)
        
        c = dolf.project(1., W)
        domainArea = dolf.assemble(c*dolf.dx)
        
        self.area = boxArea - domainArea
        
        return None
    
    def areaDerivative(self):
        """
            A list of the derivatives of the area with respect to the 
            change on the different shape parameters.
        """
        n = self.n
        dSurf = self.dSurf
        
        areaDer = []
        
        for i in range(self.dim):
            V = self.surfaceDeformation(i)
            tempDer = -dolf.dot(V,n)*dSurf(1)
            tempDer = dolf.assemble(tempDer)
            areaDer.append(tempDer)
        
        return areaDer
    