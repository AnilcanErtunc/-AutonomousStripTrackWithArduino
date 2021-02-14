from flask import Flask,request,jsonify,make_response
import xlrd
import os
import xlsxwriter




app = Flask(__name__)

cevap = "_1160"





@app.route("/", methods=['GET'])
def index():
    
    #EXCELL KISMI BAŞI*************************************************************************************************************


    dosya = os.path.dirname(os.path.realpath(__file__))    
    dosya += "\\logs.xlsx"

    dosyaNew = os.path.dirname(os.path.realpath(__file__))  
    dosyaNew += "\\log1.xlsx"

    #excell işlemleri

    loc = os.path.dirname(os.path.realpath(__file__))  

    os.system("cd "+loc+" & "+"python writer.py") #writer modülünü kerneldan çalıştırdım. 

    #Log1 yakalanacak frameyi tutuyor------------------------------------------------------


    inputWorkbook1 = xlrd.open_workbook(dosyaNew)
    inputWorksheet1 = inputWorkbook1.sheet_by_index(0)

    sayac = 0
    sayac = int(inputWorksheet1.cell_value( 0 , 0 ))   #bu kaçıncı kayda erişeceğim bilgisi 
    print(sayac)
    inputWorkbook1.release_resources()

    #OKUMA KISMI frame bilgisini logsdan alıyor------------------------------------------------------
    inputWorkbook = xlrd.open_workbook(dosya)
    inputWorksheet = inputWorkbook.sheet_by_index(0)

    print(str(inputWorksheet.cell_value(sayac , 0)))
    print(str(inputWorksheet.cell_value(sayac , 1)))

    inputWorkbook.release_resources()

    #EXCELL KISMI SONU*************************************************************************************************************
    cevap = ""
    aciRaw = inputWorksheet.cell_value(sayac , 0)
    yonRaw = str(inputWorksheet.cell_value(sayac , 1))



    if (yonRaw == "Left Curve"):
        cevap = "_0"

    elif (yonRaw == "Straight"):
        cevap = "_1"

    elif (yonRaw == "Right Curve"):
        cevap = "_2"

    cevap += "1"   #teker


    #
    # AÇI KISMI
    # 

    aciRaw = int(aciRaw * 50 ) 
    

  
    if (yonRaw == "Left Curve"):       
        cevap += str(aciRaw)

    elif (yonRaw == "Straight"):
        cevap += str(00)

    elif (yonRaw == "Right Curve"):
        aciRaw = -aciRaw
        cevap += str(aciRaw)
    
    #
    # AÇI KISMI
    # 

    sayac += 20

    return  cevap + " "
   




if __name__=="__main__":
    app.run(host="192.168.0.12",debug=True)    

#GET /  5+2 oluyordu
#GET /      '{"yon":"sag","aci":"60","teker":"off"}'