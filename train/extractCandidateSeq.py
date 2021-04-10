import re
from Bio import SeqIO
import numpy as np

def readGTF(input_file):
    """
    Extract five prime UTR information from GTF file and save as a dictionary, in which keys are transcript ID.
    """
    resUTR = {}
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip().split("\t")
            if line[0][0] != "#":
                if line[2] == "five_prime_utr":
                    chr, start, end, annotation = line[0], line[3], line[4], "".join([i.replace('"', '') for i in line[-1]])
                    transcriptID = re.search('transcript_id (.+?);', annotation).group(1)
                    if transcriptID not in resUTR.keys():
                        resUTR[transcriptID] = []
                    resUTR[transcriptID].append([chr,start,end])
    f.close()
    return resUTR

def getSeq(sequence, idx):
    upLen = idx
    downLen = len(sequence) - idx - 1
    if upLen >= 100 and downLen >= 102:
        res = sequence[(idx-100):(idx+103)]
    elif upLen >= 100 and downLen < 102:
        dollarLen = 102 - downLen
        res = sequence[(idx-100):]+"$"*dollarLen
    elif upLen < 100 and downLen >= 102:
        dollarLen = 100 - upLen
        res = "$"*dollarLen + sequence[0:(idx+103)]
    else:
        res = "$"*(100-upLen) + sequence + "$"*(102-downLen)
    return res

mainDic = "/.../PlantTISTool/PlantTIS/train/"
UTR = readGTF(input_file = mainDic + "Arabidopsis_thaliana.TAIR10.43.gtf")
cDNA = list(SeqIO.parse(mainDic + 'Arabidopsis_thaliana.TAIR10.cdna4.43.fasta', 'fasta'))
postives = open("TAIR10.fasta", "w") # output

for record in cDNA:
    name, sequence = record.id, str(record.seq)
    if name in UTR.keys():
        curUTR = UTR[name]
        curLength = sum([int(i[2])-int(i[1])+1 for i in curUTR]) # the length of five prime UTR
        motif = sequence[curLength:(curLength+3)]
        index = [match.start() for match in re.finditer(re.escape(motif), sequence)] # find all motifs
        for idx in index:
            curSeq = getSeq(sequence = sequence, idx = idx)
            if idx == curLength:
                ifPositive = "Y"
            else:
                ifPositive = "N"
            postives.write(">" + name + "_" + str(idx) + "_" + ifPositive + "\n")
            postives.write(curSeq + "\n")
        #print (name + "\n")
postives.close()

# convert to numpy array
candidates = list(SeqIO.parse('TAIR10.fasta', 'fasta'))
seqIDs = []
sequences = []

for record in candidates:
    name, sequence = record.id, str(record.seq)
    seqIDs.append(name)
    sequences.append(sequence)

seqArray = np.array(sequences)
np.save("TAIR10.npy", seqArray)
