
import xlrd
import os
import xlsxwriter



dosyaNew = os.path.dirname(os.path.realpath(__file__))  
dosyaNew += "\\log1.xlsx"


#OKUMA KISMI değiştireceğim kayıt------------------------------------------------------

inputWorkbook1 = xlrd.open_workbook(dosyaNew)
inputWorksheet1 = inputWorkbook1.sheet_by_index(0)

sayac = 0
sayac = int(inputWorksheet1.cell_value( 0 , 0 ))   #bu kaçıncı kayda erişeceğim bilgisi 
inputWorkbook1.release_resources()


#YAZMA KISMI---------------------------------------------------- sayaca 30 ekleyecek frame yakalamak için 
if(sayac <= 500): #video sonu
    sayac = sayac + 22

import xlsxwriter


workbook = xlsxwriter.Workbook('log1.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write_number(0,0,sayac )


workbook.close()




