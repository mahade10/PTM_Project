import re
def window_sequence(infile,outfile,residue)
    with open(infile,"r") as fin, open(outfile,"w") as fout:
        #residue="S"
        wind_size=10
        pssm_count=0
        for line in fin:
            if line.startswith(">"):
                pssm_count+=1
                protein = line[1:7]
                #residue=line[-2];
                positive=[int(s[1:]) for s in re.findall(r'#\d+', line)]
            else:
                pssm="pssm"+str(pssm_count)
                count=0
                for ch in line:
                    count+=1
                    if ch==residue:
                        if count<=wind_size:
                            motif=line[0:count+wind_size].rjust(21,'-')
                        elif count+wind_size>=len(line):
                            motif = line[count-1-wind_size:-1].ljust(21, '-')
                        else:
                            motif=line[count-1-wind_size:count+wind_size]
                        if count in positive:
                            cls="+1"
                        else:
                            cls="-1"
                        println=motif+"\t"+protein+"\t"+str(count)+"\t"+residue+"\t"+cls+"\t"+pssm
                        print(println,file=fout)
