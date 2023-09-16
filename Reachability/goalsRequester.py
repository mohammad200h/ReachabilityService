#!/usr/bin/env python

from .com import Client

class GoalsRequester(Client):
    def __init__(self):
        super().__init__()
        
        self.goals = {
            "ff":None,
            "mf":None,
            "rf":None,
            "th":None
        }
    
    def send_pregrasp_and_wait_for_result(self,pregrasp):
        self.goals = self.send(pregrasp)
    
    def get_update(self):
        return self.goals