dict={}
with open('tabPFN.txt',"r") as fidR1:
    for i in fidR1.readlines():
        key=i.split('\t')[0]
        value=i.split('\t')[1:]
        dict[key]=value
fidW = open('ml_tabPFN.txt', 'w')
with open('ml_list.txt',"r") as fidR2:
    for i in fidR2.readlines():
        if i.strip() in dict.keys():
            writeline = i.strip() + '\t' + '\t'.join(dict[i.strip()])
        else:
            writeline=i
        fidW.write(writeline)
fidW.close()









#########################################
# ftxt="11.txt"
# fidW = open('11_v1.txt', 'w')
# with open(ftxt,"r") as fidR:
#     for i in fidR.readlines():
#         writeline=i.replace(' ','\t')
#         fidW.write(writeline)
# fidW.close()