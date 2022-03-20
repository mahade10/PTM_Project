# Save encoding in the same file.

#import re
#from collections import Counter
from descproteins import *
from pubscripts import save_file

tags = ["Train"]
residues = ["Y"]
for residue in residues:
    for tag in tags:
        #fname = "dataset/TrainW21Y_2299.txt"
        fname = "dataset/" + tag + "W21" + residue + ".txt"
        fout = "Encoding_result/" + tag + residue + "_Encoding.txt"
        ffeature = "Encoding_result/" + tag + residue + "_Features.txt"
        with open(fname, 'r') as fin, open(fout, 'w') as f, open(ffeature, 'w') as ff:
            flag = 0
            for line in fin:
                fastas = []
                words = line.split("\t")
                name = words[1]
                sequence = words[0]
                label = words[4]
                fastas.append([name, sequence, label])
                myOrder = "ACDEFGHIKLMNPQRSTVWY"
                filePath = "data/"
                kw = {'path': filePath, 'order': myOrder, 'type': 'Protein'}
                choices = ['CKSAAP','binary','PAAC','AAINDEX']
                #choices = ['AAC', 'EAAC', 'CKSAAP', 'DPC', 'DDE', 'TPC', 'binary','GAAC', 'EGAAC', 'CKSAAGP', 'GDPC', 'GTPC','AAINDEX','ZSCALE', 'BLOSUM62','CTDC', 'CTDT', 'CTDD']
                #   'APAAC', 'PAAC','NMBroto', 'Moran', 'Geary','CTriad', 'KSCTriad', 'KNNprotein','KNNpeptide',
                #   'PSSM', 'SSEC', 'SSEB', 'Disorder', 'DisorderC', 'DisorderB', 'ASA', 'TA']
                #choices = ['AAC', 'EAAC', 'CKSAAP', 'DPC', 'DDE', 'TPC', 'binary',
                    #'GAAC', 'EGAAC', 'CKSAAGP', 'GDPC', 'GTPC',
                    #'AAINDEX', 'ZSCALE', 'BLOSUM62',
                    #'NMBroto', 'Moran', 'Geary',
                    #'CTDC', 'CTDT', 'CTDD',
                    #'CTriad', 'KSCTriad',
                    #'SOCNumber', 'QSOrder',
                    #'PAAC', 'APAAC',
                    #'KNNprotein', 'KNNpeptide',
                    #'PSSM', 'SSEC', 'SSEB', 'Disorder', 'DisorderC', 'DisorderB', 'ASA', 'TA'
                    #]
                if flag == 0:
                    ff.write("Class\n")
                f.write('%s' % label)
                for desc in choices:
                    cmd = desc + '.' + desc + '(fastas, **kw)'
                   # print(cmd)
                    encodings = eval(cmd)
                    if flag == 0:
                        fln = encodings[0]
                        ff.write("%s-%d: " % (desc, len(fln)-2))
                        for k in range(2, len(fln)):
                            ff.write(',%s' % fln[k])
                        ff.write('\n')
                    ln = encodings[1:]
                    #print(encodings[0])
                    ln = ln[0]
                    for i in range(2, len(ln)):
                        f.write(',%s' % ln[i])
                   # fout = "Encoding_result/"+residue+"_Encoding/"+label+residue+desc+".csv"
                    #save_file.save_file(encodings, "csv", fout)
                f.write('\n')
                flag = 1
