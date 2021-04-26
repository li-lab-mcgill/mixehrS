# the original code also provides a version with 3-digit ICD-9 code
# but we only use full ICD-9 code here.
import sys
import cPickle as pickle
from datetime import datetime

def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4] + '.' + dxStr[4:]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3] + '.' + dxStr[3:]
        else:
            return dxStr

if __name__ == '__main__':
    admissionFile = sys.argv[1]
    diagnosisFile = sys.argv[2]
    outFile = sys.argv[3]

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')
    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) # The origianl code also provides a 3-digit ICD-9 digits.
        if admId in admDxMap: 
            admDxMap[admId].append(dxStr)
        else: 
            admDxMap[admId] = [dxStr]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    pidSeqMap_3digit = {}
    for pid, admIdList in pidAdmMap.iteritems():
        if len(admIdList) < 2:
            continue
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
        pidSeqMap[pid] = sortedList

    print('Building pids, dates, strSeqs')
    pids = []
    dates = []
    seqs = []
    for pid, visits in pidSeqMap.iteritems():
        pids.append(pid)
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)
    
    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    pickle.dump(pids, open(outFile+'.pids', 'wb'), -1)
    pickle.dump(dates, open(outFile+'.dates', 'wb'), -1)
    pickle.dump(newSeqs, open(outFile+'.seqs', 'wb'), -1)
    pickle.dump(types, open(outFile+'.types', 'wb'), -1)
