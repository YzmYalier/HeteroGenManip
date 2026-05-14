import numpy as np
from isaacsim.core.api import World
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema

class CollisionGroup:
    def __init__(self, world:World,helper_path:list=None,garment:bool=True,collide_with_garment:bool=True,collide_with_robot:bool=True, garment_with_robot:bool=True):
        # get stage
        self.stage = world.stage
        
        # ----------define garment_collision_group---------- #
        # path
        self.garment_group_path = "/World/Collision_Group/garment_group"
        # collision_group
        self.garment_group = UsdPhysics.CollisionGroup.Define(self.stage, self.garment_group_path)
        # filter(define which group can't collide with current group)
        self.filter_garment = self.garment_group.CreateFilteredGroupsRel()
        # includer(push object in the group)
        self.collectionAPI_garment = Usd.CollectionAPI.Apply(self.filter_garment.GetPrim(), "colliders")
        
        if helper_path is not None:
            # ----------define helper_collision_group---------- #
            self.helper_group_path = "/World/Collision_Group/helper_group"
            self.helper_group = UsdPhysics.CollisionGroup.Define(self.stage, self.helper_group_path)
            self.filter_helper = self.helper_group.CreateFilteredGroupsRel()
            self.collectionAPI_helper = Usd.CollectionAPI.Apply(self.filter_helper.GetPrim(), "colliders")

        # ----------define robot_collision_group---------- #
        self.robot_group_path = "/World/Collision_Group/robot_group"
        self.robot_group = UsdPhysics.CollisionGroup.Define(self.stage, self.robot_group_path)
        self.filter_robot = self.robot_group.CreateFilteredGroupsRel()
        self.collectionAPI_robot = Usd.CollectionAPI.Apply(self.filter_robot.GetPrim(), "colliders")
        
        # push objects to different group
        if garment:
            self.collectionAPI_garment.CreateIncludesRel().AddTarget("/World/Garment")
        else:
            self.collectionAPI_garment.CreateIncludesRel().AddTarget("/World/Deformable")
        if helper_path is not None:
            for helper in helper_path:
                self.collectionAPI_helper.CreateIncludesRel().AddTarget(helper)
        self.collectionAPI_robot.CreateIncludesRel().AddTarget("/World/DexLeft")
        self.collectionAPI_robot.CreateIncludesRel().AddTarget("/World/DexRight")
        
        if helper_path is not None:
        # allocate the filter attribute of different groups
            if not collide_with_garment:
                self.filter_helper.AddTarget(self.garment_group_path)
            if not collide_with_robot:
                self.filter_helper.AddTarget(self.robot_group_path)
        if not garment_with_robot:
            self.filter_robot.AddTarget(self.garment_group_path)
        
        
    def add_collision(self,group_path:str,target:list,garment:bool=False,helper:bool=False,robot:bool=False):
        self.group_path = f"/World/Collision_Group/{group_path}_group"
        self.group = UsdPhysics.CollisionGroup.Define(self.stage, self.group_path)
        self.filter = self.group.CreateFilteredGroupsRel()
        self.collectionAPI= Usd.CollectionAPI.Apply(self.filter.GetPrim(), "colliders")
        for target_path in target:
            self.collectionAPI.CreateIncludesRel().AddTarget(target_path)

        if not garment:
            self.filter.AddTarget(self.garment_group_path)
        if not robot:
            self.filter.AddTarget(self.robot_group_path)
        if not helper:
            self.filter.AddTarget(self.helper_group_path)