import os

import os,sys,json
import numpy as np
from elasticsearch import Elasticsearch


class LinUCB:
	def __init__ (self,hparams,user):

		self.alpha = hparams['explore_rate']
		self.d = hparams['context_dim']
		self.r1 = 0.6
		self.r0 = -16
		self.Aa = {}
		self.AaI = {}
		self.ba = {}
		self.a_max = 0
		self.theta = {}
		self.x = None
		self.xT = None


	def init_actions(self,actions):

		for action in actions:
			self.Aa[action] = np.identity(self.d)
			self.ba[action] = np.zeros((self.d,1))

			self.AaI[action] = np.identity(self.d)
			self.theta[action] = np.zeros((self.d,1))


	def update(self,reward):

		r = reward

		self.Aa[self.a_max] += np.dot(self.x,self.xT)
		self.ba[self.a_max] += r * self.x
		self.AaI[self.a_max] = np.linalg.inv(self.Aa[a_max])
		self.theta[self.a_max] = np.dot(self.AaI[self.a_max],self.ba[self.a_max])


	def recommend(self,timestep,context_features,actions):
		xaT = np.array([context_features])
		xa = np.transport(xaT)

		AaI_tmp = np.array([self.AaI[action]] for action in actions)
		theta_tmp = np.array([self.thetha[action]] for action in actions)
		doc_max = actions[np.argmax(np.dot(xaT,theta_tmp)+ self.alpha * np.sqrt(np.dot(np.dot(xaT,AaI_tmp),xa)))]

		self.x = xa
		self.xT = xaT

		self.a_max = doc_max

		return self.a_max
