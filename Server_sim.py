# -*- coding: utf-8 -*-
# Program for server simulation 
# COMP9334 System Capacity
# Session 1, 2017
# Author Yihan Xiao z5099956
# This program works with a python 
# open source discrete event package Simpy

import simpy as si
import numpy as np
import pandas as pd
import math as mt
import random as rd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import csv
from collections import namedtuple

# test file
jobEntry = namedtuple('jobEntry','index arrival_time service_time')
job1 = jobEntry(1, 1, 2.1)
job2 = jobEntry(2, 2, 3.3)
job3 = jobEntry(3, 3, 1.1)
job4 = jobEntry(4, 5, 0.5)
job5 = jobEntry(5, 15,1.7)

timeStamp = namedtuple('timeStamp','timeNow remainServ')

job_arrival_list = [job1,job2,job3,job4,job5]

def working_freq(num_of_server):
    p = 2000.0 / num_of_server
    f = 1.25 + 0.31*( (p/200.0) - 1 )
    
    return f

# Arrival time
def arrival_time(N, freq):
    next_arrival_time = 0
    next_service_time = 0
    
    for i in range(N):
        # a1k -- exponentially distributed with a mean arrival rate 7.2
        a1k = rd.expovariate(7.2)
        # a2k -- uniformly distribution in range [0.75,1.17]
        a2k = rd.uniform(0.75,1.17)
        
        next_arrival_time += a2k * a1k
        
        next_service_time = service_time(freq)
        
    return next_arrival_time, next_service_time

# Service time
def service_time(freq):
    alpha1 = 0.43
    alpha2 = 0.98
    beta = 0.86
    
    gama = (1-beta) / (mt.pow(alpha2,1-beta) - mt.pow(alpha1,1-beta))
    lower_bond = mt.pow(alpha1, 1-beta) * gama / (1-beta)
    # random number
    X = rd.uniform(0,1)
    Y = mt.pow((X+lower_bond) * (1-beta)/gama, 1.0/(1-beta))
    
    return Y/freq


class Job(object):
    
    def __init__(self, index, start_time, service_time):
        self.index = index
        self.start_time = start_time
        self.service_time = service_time
        self.service_time_ori = service_time
        self.finish_time = float('inf')
        self.response_time = float('inf')
        self.time_stamp = []

#-------------- Input jobs from a list. For test only------------
def Arrival_fromList(env, jobs, server):
    
    while len(jobs) > 0:
        job = jobs.pop(0)
        new_job = Job(job.index, job.arrival_time, job.service_time)
        new_stamp = timeStamp(new_job.start_time, new_job.service_time)
        new_job.time_stamp.append(new_stamp)
        server.isNext = True
        
        yield env.timeout(new_job.start_time - env.now)
        #print('Ready to send job at ', env.now)
         # interrupt and update job list
#        print('Job%d arrives at %f need %f' %(new_job.index, env.now, new_job.service_time))
#        print('===========================')
        if server.isRuning == True:
            server.process.interrupt()
        # insert job after updating
        server.updateJobs(new_job = new_job)
        server.process = env.process(server.serverRun())
        
    server.stop = True
    
def Arrival_fromTime(env, num_of_server, num_of_job, server, freq, isJobLimited = False):
    index = 1
   
    while True:
        time_interval, service_time = arrival_time(num_of_server, freq)
        
        new_job = Job(index, env.now + time_interval, service_time)
        new_stamp = timeStamp(new_job.start_time, new_job.service_time)
        new_job.time_stamp.append(new_stamp)
        
        yield env.timeout(time_interval)
#        print('Ready to send job at ', env.now)
         # interrupt and update job list
#        print('Job%d arrives at %f need %f' %(new_job.index, env.now, new_job.service_time))
#        print('===========================')
        if server.isRuning == True:
            server.process.interrupt()
        # insert job after updating
        server.updateJobs(new_job = new_job)
        server.process = env.process(server.serverRun())
        
        index += 1
        if isJobLimited == True:
            if index > num_of_job: break
        else:
            if len(server.job_result_list) >= num_of_job: break
           
class Server(object):
    
    def __init__(self, env, num_of_server, num_of_job):
        self.env = env
        self.job_list = []
        self.job_result_list = []
        self.NoS = num_of_server
        self.NoJ = num_of_job
        self.isRuning = False
        self.next_departrue = float('inf')
        self.last_time = 0
        self.process = None
       
    def updateJobs(self, over_index = None, new_job = None):
        if self.job_list != []:
            
            speed = 1.0 / len(self.job_list)
    
            for job in self.job_list:
                if job.index != over_index:
                    job.service_time -= (self.env.now - self.last_time) * speed
                    new_stamp = timeStamp(self.env.now, job.service_time)
                    job.time_stamp.append(new_stamp)
                else:
                    job.service_time = 0
                    new_stamp = timeStamp(self.env.now, job.service_time)
                    job.time_stamp.append(new_stamp)
        
        if new_job:
            self.job_list.append(new_job)
            
        self.last_time = self.env.now
        
    def serverRun(self):
        
        while self.job_list != []:
            if len(self.job_result_list) >= self.NoJ:
                break
           
            self.isRuning = True
#            print('Running!----------------', self.env.now)
            # find next departure time
            job_ready = min(self.job_list, key = lambda x : x.service_time)
#            print('Job to go:', job_ready.index, job_ready.service_time)
            self.next_departrue = job_ready.service_time * len(self.job_list)
#            print('Next departure could be', self.next_departrue)
            try:              
                # if successfully finish a job
#                print('try pass',self.next_departrue - self.env.now)
                yield self.env.timeout(self.next_departrue)
#                print('time pass, now', self.env.now)
                self.updateJobs(over_index = job_ready.index)
                # record this job
                job_ready.finish_time = self.env.now
                job_ready.response_time = job_ready.finish_time - job_ready.start_time
                
          
                self.job_result_list.append(job_ready)
                # delete it from the job list
                self.job_list.remove(job_ready)
#                print('***Job%d start at %f need %f finish at %f res %f' % 
#                      (job_ready.index, job_ready.start_time, job_ready.service_time_ori, 
#                       job_ready.finish_time, job_ready.response_time))
                
            
            except si.Interrupt:
                self.isRuning = False
                # shutdown the loop
                break
#                print('Interrupted')

            
        # process ends, stop runing
        self.isRuning = False
        
def simulation(num_of_server, num_of_job, mode=1, mean=1, w=False, isJobLimited=False):
    assert mean >= 0
    
    freq = working_freq(num_of_server)
    
    env = si.Environment()
    myServer = Server(env, num_of_server, num_of_job) 
    if mode:
        env.process(Arrival_fromTime(env, num_of_server, num_of_job, myServer, freq, isJobLimited))    
    else:
        env.process(Arrival_fromList(env, job_arrival_list, myServer))    

    env.run()  

    myServer.job_result_list.sort(key= lambda x : x.index)
    analyse_data_ori = pd.DataFrame([x.response_time for x in myServer.job_result_list])
    analyse_data = analyse_data_ori.copy()
    if mean:
        for i in range(num_of_job):
            if i < mean:
                analyse_data[0][i] = analyse_data_ori[0][i:2*i+1].mean()
            elif num_of_job-i < mean:
                analyse_data[0][i] = analyse_data_ori[0][i-mean:].mean()
            else:    
                analyse_data[0][i] = analyse_data_ori[0][i-mean:i+mean].mean()
    
    # write result to csv file
    if w == True:
        header = ['Job Index','Arrival','Service','Departure','Response']
        with open('result.csv', 'w', newline='') as file:
            job_writer = csv.writer(file, dialect='excel')
            job_writer.writerow(header)
            for job in myServer.job_result_list:
                job_writer.writerow(
                        [job.index,job.start_time,
                         job.service_time_ori,job.finish_time,
                         job.response_time]
                        )
   
#    for job in myServer.job_result_list:
#        plt.plot([x.timeNow for x in job.time_stamp],
#                 [y.remainServ for y in job.time_stamp])
#    plt.show()          
    
#    plt.plot([x.index for x in mean_list],
#             [x.start_time for x in mean_list])
#    plt.plot([x.index for x in mean_list],
#             [x.service_time_ori for x in mean_list])
    return analyse_data

# cal mean response time
def mean_response(L, start):
    sumData = 0
    index = start
    while index < len(L):
        sumData += L[index]
        index += 1
    return sumData / (len(L)-start)
    
def multi_sim(num_repeat, num_of_server, num_of_job, mode=1, mean=1, w=False):
    data_set = pd.DataFrame()
    
    data_mean_reponse = []
    if(mean == 'AUTO'): mean = num_of_job // 10
    
    for i in range(num_repeat):
        rd.seed(i*200)
        if len(data_set) == 0:
            data_set = simulation(num_of_server, num_of_job, mode, mean).copy()
            
            data_mean_reponse.append(mean_response(data_set[0],1000))            
        else:
            data_set = simulation(num_of_server, num_of_job, mode, mean)
            
            data_mean_reponse.append(mean_response(data_set[0],1000))            

    if num_repeat == 1:
        plt.plot(data_set)
        plt.xlabel('Server=%d  w=%d' % (num_of_server,mean))
        plt.show()
    
    # write result to csv file
    if w == True:
        header = ['Server Number', 'Mean Response']
        with open('mean.csv', 'w', newline='') as file:
            job_writer = csv.writer(file, dialect='excel')
            job_writer.writerow(header)
            for mRes in data_mean_reponse:
                job_writer.writerow(
                        [num_of_server, mRes]
                        )
    return data_mean_reponse

# main function
mean1 = np.array(multi_sim(20, 8, 5000, mode=1, mean=0, w=False))
mean2 = np.array(multi_sim(20, 7, 5000, mode=1, mean=0, w=False))
mean_diff = mean1 - mean2
df = mean_diff.size - 1
SE = np.std(mean_diff) / np.sqrt(mean_diff.size)
a,b = ss.t.interval(0.95, df=df, loc=np.mean(mean_diff), scale=SE)
print(a,b)
