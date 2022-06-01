import pygmsh as pg
import math
import os
import dolfin as dolf
import numpy as np
import sympy as sp

class GivenAirfoil:
    """
        An airfoil which has its points specified. (dim = 0) Not every field is created and not 
        every function is implemented. It is best to use it only for calculating flow around it.
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
    """
    
    def __init__(self, resolution, outerGeometry, airfoilPoints):
        """
            Initialising arguments:
                resolution: The resolution of the airfoil ie. the number of points of the airfoil.
                outerGeometry: The geometry of the outer bounding box.
                airfoilPoints: A list of the (x,y) surface points of the airfoil. It is advisable to 
                    have rougly resoultion number of points on the airfoil.
        """
        # Setting the variables
        self.resolution = resolution
        self.outerGeometry = outerGeometry
        self.airfoilPoints = airfoilPoints
        # The number of parameters
        self.dim = 0
        
        meshName = "mesh/circle_mesh_"+str(resolution)
        
        # Creating the mesh
        self.createMesh(meshName)
        
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
    
    def createMesh(self, meshName, saveMesh=False):
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
                mesh: The mesh of the domain.
        """
        geom = pg.Geometry()
        outerCircumference = (self.outerGeometry["xFrontLength"] + self.outerGeometry["xBackLength"])*2.0 + \
                              self.outerGeometry["yHalfLength"]*4.0
        
        xMin = self.airfoilPoints[0][0]
        xMax = self.airfoilPoints[0][0]
        yMin = self.airfoilPoints[0][1]
        yMax = self.airfoilPoints[0][1]
        
        
        # Points on the airfoil
        airfoilPoints = []
        for points in self.airfoilPoints:
            (xAirfoil,yAirfoil) = points[0], points[1]
            airfoilPoints.append((xAirfoil,yAirfoil,0.0))
            
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
    
    