import pygmsh as pg
import math
import os
import dolfin as dolf
import numpy as np
import sympy as sp

class CircleAirfoil:
    """
        A circular airfoil. (dim = 1)
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
    
    def __init__(self, resolution, outerGeometry, circleParameters):
        """
            Initialising arguments:
                resolution: The resolution of the airfoil ie. the number of points of the airfoil.
                outerGeometry: The geometry of the outer bounding box.
                circleParamters: A list containing one element the radius of the circle.
        """
        # Setting the variables
        self.resolution = resolution
        self.outerGeometry = outerGeometry
        self.rCircle = circleParameters[0]
        # The number of parameters
        self.dim = 1
        
        meshName = "mesh/circle_mesh_"+str(resolution)
        
        # Creating the mesh
        self.createCircleMesh(meshName)
        
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
    
    def createCircleMesh(self, meshName, saveMesh=False):
        """
            Make a mesh around a Joukowsky airfoil. The mesh is also saved 
            as a .xml file if saveMesh is True. It is important not to create two meshes at 
            the same time with the same name.
            Input argument:
                meshName: The name of the mesh. If saveMesh is True the mesh is saved as meshName.xml.
            Optional argument:
                saveMesh: If it is True the mesh is saved.
            Created fields:
                deltax: The width of the airfoil.
                deltay: The thickness of the airfoil.
                airfoilPoints: The list of (x,y) coordinates of the vertices of the airfoil.
                mesh: The mesh of the domain.
        """
        geom = pg.Geometry()
        outerCircumference = (self.outerGeometry["xFrontLength"] + self.outerGeometry["xBackLength"])*2.0 + \
                              self.outerGeometry["yHalfLength"]*4.0
        
        
        # Angles of the airfoil
        airfoilAngles = np.linspace(0., 2.0*math.pi, self.resolution, endpoint=False)
        
        # Points on the airfoil
        airfoilPoints = []
        self.airfoilPoints = []
        for angle in airfoilAngles:
            (xAirfoil,yAirfoil) = (self.rCircle*math.cos(angle), self.rCircle*math.sin(angle))
            self.airfoilPoints.append((xAirfoil,yAirfoil))
            airfoilPoints.append((xAirfoil,yAirfoil,0.0))
            
        
        
        # Thickness and length
        self.deltax = 2.*self.rCircle
        self.deltay = 2.*self.rCircle
        
        # The airfoil surface.
        # The characteristic length sould be larger than the actual distance between the points.
        airfoil = geom.add_polygon(airfoilPoints, outerCircumference, make_surface = False)
        
        bBoxPoints = airfoilPoints
        bBox = airfoil
        # Bounding boxes:
        # Small bounding box
        bBoxPoints = [[-2.*self.rCircle,  2.*self.rCircle, 0.0], \
                      [-2.*self.rCircle, -2.*self.rCircle, 0.0], \
                      [ 2.*self.rCircle, -2.*self.rCircle, 0.0], \
                      [ 2.*self.rCircle,  2.*self.rCircle, 0.0]]
        bBox = geom.add_polygon(bBoxPoints, 8.*self.rCircle/self.resolution, holes=[bBox])
        # Middle bounding box
        bBoxPoints = [[-0.5*(self.outerGeometry["xFrontLength"]+2.*self.rCircle), \
                        0.5*(self.outerGeometry["yHalfLength"]+2.*self.rCircle), 0.0], \
                    [-0.5*(self.outerGeometry["xFrontLength"]+2.*self.rCircle), \
                        -0.5*(self.outerGeometry["yHalfLength"]+2.*self.rCircle), 0.0], \
                    [ 0.5*(self.outerGeometry["xBackLength"]+2.*self.rCircle), \
                        -0.5*(self.outerGeometry["yHalfLength"]+2.*self.rCircle), 0.0], \
                    [ 0.5*(self.outerGeometry["xBackLength"]+2.*self.rCircle), \
                        0.5*(self.outerGeometry["yHalfLength"]+2.*self.rCircle), 0.0]]
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

    
    def inverseCircleTransform(self, coords):
        """
            Get the theta angle from the coordinates of the airfoil point.
        """
        theta = math.atan2(coords[1], coords[0])
        if(theta < 0.):
            theta += 2.*math.pi
        
        return theta
    
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
                    theta = self.inverseCircleTransform((v.x(0), v.x(1)))
                    (dX, dY) = math.cos(theta), math.sin(theta)
                    
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
    