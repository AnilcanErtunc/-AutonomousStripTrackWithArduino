# Gerekli olan kütüphaneleri import ediyorum
import cv2
import numpy as np
import os

from matplotlib import pyplot as plt, cm, colors

import xlsxwriter

#####################################EXCEL TANIMLAR
# =============================================================================
# planWorkbook = xlsxwriter.Workbook("logs.xlsx")
# planSheet = planWorkbook.add_worksheet("yonler")
# derece = ''
# yon = ''
# 
# sayac = 0
# curveRad = ''
# curveDir = ''
#  print("!!!!excell modunda açıldı!!!!")
# 
# =============================================================================

########################################################################Excelli Sıfırla
planWorkbookExit = xlsxwriter.Workbook("log1.xlsx")
planSheetExit = planWorkbookExit.add_worksheet()
zero = 0

planSheetExit.write( 0 , 0 , zero )
print("log1 sıfırlandı!!")
planWorkbookExit.close()


########################################################################Excelli Sıfırla
####################################EXCEL TANIMLAR BİTİŞ


# Metre piksel dönüşümünü tutacak değişkenleri tanımladığım kısım
#y boyutunda piksel başına metre
ym_per_pix = 30 / 720
# Standart şerit genişliği 3,7 metredir ve piksel cinsinden şerit genişliğine bölünür.
# çerçeve yüksekliği ile karıştırılmaması için yaklaşık 720 piksel olarak hesaplanmıştır
xm_per_pix = 3.7 / 720

# Mevcut çalışma dizininin yolunu alıyorum
CWD_PATH = os.getcwd()
######## GÖRÜNTÜ İŞLEMEYİ GERÇEKLEŞTİREN FONKSİYONLARI BAŞLADIĞI KISIM #########################

#### Giriş Görüntüsünü Okumak için FoNKSİYONLAR ###################################
def readVideo():

    # giriş videosunu okunup GirisGoruntu atılması
    GirisGoruntu = cv2.VideoCapture(os.path.join(CWD_PATH, 'drive.mp4'))
    
    return GirisGoruntu

################################################################################



################################################################################
####  Başladığı Yer GÖRÜNTÜ İŞLEME İŞLEVİ #########################################
def onIsleme(GirisGoruntu):
    # Beyaz şerit çizgilerini filtrelemek için HLS renk filtrelemesi uygulayın
    hls = cv2.cvtColor(GirisGoruntu, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 160, 10])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(GirisGoruntu, lower_white, upper_white)
    hls_sonuc = cv2.bitwise_and(GirisGoruntu, GirisGoruntu, mask = mask)

     # Görüntüyü gri tonlamaya dönüştürüldüğü, görüntünün bulanıklaştırıldığı ve kenarlarının çıkarıldığı kısım 
    gray = cv2.cvtColor(hls_sonuc, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh,(3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)
    # #İşlenmiş görüntüler
    # cv2.imshow("Image", GirisGoruntu)
    # cv2.imshow("HLS Filtered", hls_sonuc)
    # cv2.imshow("Grayscale", gray)
    # cv2.imshow("Thresholded", thresh)
    # cv2.imshow("Blurred", blur)
    # cv2.imshow("Canny Edges", canny)

    return image,hls_sonuc, gray, thresh, blur, canny
#### Görüntü İşleminin Bittiği Yer  ###########################################
################################################################################


################################################################################
#### Perspective warp uygulandığı kısmın başladığı yer ################################
def PerspectiveWrp(GirisGoruntu):

    # Resimlerin boyutunu alıyoruz
    img_size = (GirisGoruntu.shape[1], GirisGoruntu.shape[0])
    # print(img_size)
    # Eğriltilcek perspectiv noktaları
    kyk = np.float32([[590, 440],
                      [690, 440],
                      [200, 640],
                      [1000, 640]])

    # Gösterilecek pencere(Window slide da ki pencereler)
    hdf = np.float32([[200, 0],
                      [1200, 0],
                      [200, 710],
                      [1200, 710]])

    # Kuş bakışı pencere için görüntüyü çarpıtıldığı yer
    matrix = cv2.getPerspectiveTransform(kyk,  hdf)
    # Son pencere için görüntüyü açmak için ters matris
    min_Pencere = cv2.getPerspectiveTransform( hdf, kyk)
    kusbakisi = cv2.warpPerspective(GirisGoruntu, matrix, img_size)

    global kusbakisim
    kusbakisim = cv2.warpPerspective(GirisGoruntu, matrix, img_size)
    # Kuş bakışı pencere boyutlarını alındığı kısım
    height, width = kusbakisi.shape[:2]
    # print(height,width)
    # Sol ve sağ şeritleri ayırmak için kuş bakışı görünümünü 2 yarıya bölün
    kusbakasiSol  = kusbakisi[0:height, 0:width // 2]
    kusbakasiSag = kusbakisi[0:height, width // 2:width]

    # Kuş bakışı imagelerin görünümü
    # cv2.imshow("Birdseye" , birdseye)
    # cv2.imshow("Birdseye Left" , kusbakasiSol)
    # cv2.imshow("Birdseye Right", kusbakasiSag)
    return kusbakisi, kusbakasiSol, kusbakasiSag,  min_Pencere
#### Perspective warp uygulandığı kısmın bittiği yer ##################################
################################################################################



################################################################################
#### HİSTOGRAMINI ÇİZİM İÇİN Warped image Başladığı Yer ####################
def Histogram(GirisGoruntu):
    #Görüntünün alt yarısın histogramını alır
    histogram = np.sum(GirisGoruntu[GirisGoruntu.shape[0] // 2:, :], axis = 0)
    #histogram.shape[0]/2=640 yani x koordinatına göre orta_nokta değeri=640
    orta_nokta = np.int(histogram.shape[0] / 2)
    solxKaynak = np.argmax(histogram[:orta_nokta])
    sagxKaynak = np.argmax(histogram[orta_nokta:]) + orta_nokta

    plt.xlabel("Image X Coordinates")
    # plt.ylabel("Number of White Pixels")
    
    # sol ve sağ şeritlerin histogramını hesaplamak için x kordinatlarını döndürünldüğü kısım
    # plt.show(solxKaynak,sagxKaynak)
    # piksel cinsinden şerit genişliği
    return histogram, solxKaynak, sagxKaynak
#### HİSTOGRAMINI ÇİZİM İÇİN Warped image BİTTİĞİ Yer  ######################
################################################################################


################################################################################
#### EĞRİLERİ TESPİT ETMEK İÇİN SLİDE WİNDOW YÖNTEMİNİ UYGULAYANDIĞI KISMIN BAŞLADIĞI YER ######################
def Slide_Window_Search(binary_warped, histogram):
     
    # Histogram bilgisini kullanarak sol ve sağ şerit çizgilerinin başlangıcını bulunması
    #histogram.shape[0]/2 e eksenine göre img de ki orta noktayı veriyor buda 360 eşittir
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    orta_nokta = np.int(histogram.shape[0] / 2)
    ana_solx = np.argmax(histogram[:orta_nokta])
    ana_sagx = np.argmax(histogram[orta_nokta:]) + orta_nokta
    # Toplam 9 pencere kullanılacak
    npencere = 9
    #Pencere yüksekliğini ayarlıyoruz
    pencere_yukseklik = np.int(binary_warped.shape[0] / npencere)
    nonzero = binary_warped.nonzero()
    #Görüntüdeki sıfır olmayan tüm piksellerin x ve y konumlarını tanımlanıyor
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
 
    sol_egri =ana_solx
    sagx_egri = ana_sagx
    #pencere genişliği
    margin = 100
    #pencere oluşturmak için gereken minümum piksel boyutu
    minpix = 50
    sol_serit_inds = []
    sag_serit_inds = []
    
    ####  Pencereler arasında yineleme yapmak ve şerit çizgilerini aramak için döngü yapıldığı yer#####
    for pencere in range(npencere):
        pencere_y_dusuk = binary_warped.shape[0] - (pencere+ 1) * pencere_yukseklik
        pencere_y_yuksek = binary_warped.shape[0] - pencere* pencere_yukseklik
        #x ve y'de (sağda ve solda) pencere sınırlarını belireniyor
        pencer_xsol_dusuk =  sol_egri - margin
        pencere_xsol_yuksek =  sol_egri + margin
        pencere_xsag_dusuk = sagx_egri - margin
        pencere_xsag_yuksek = sagx_egri + margin
        #Görüntü üzerine pencerenin çizildiği yer
        cv2.rectangle(out_img, (pencer_xsol_dusuk, pencere_y_dusuk), (pencere_xsol_yuksek, pencere_y_yuksek),
        (0,255,0), 2)
        cv2.rectangle(out_img, (pencere_xsag_dusuk,pencere_y_dusuk), (pencere_xsag_yuksek,pencere_y_yuksek),
        (0,255,0), 2)
        iyi_sol_inds = ((nonzeroy >= pencere_y_dusuk) & (nonzeroy < pencere_y_yuksek) &
        (nonzerox >= pencer_xsol_dusuk) &  (nonzerox < pencere_xsol_yuksek)).nonzero()[0]
        iyi_sag_inds = ((nonzeroy >= pencere_y_dusuk) & (nonzeroy < pencere_y_yuksek) &
        (nonzerox >= pencere_xsag_dusuk) &  (nonzerox < pencere_xsag_yuksek)).nonzero()[0]
        sol_serit_inds.append(iyi_sol_inds)
        sag_serit_inds.append(iyi_sag_inds)
        if len(iyi_sol_inds) > minpix:
           solx_egri = np.int(np.mean(nonzerox[iyi_sol_inds]))
        if len(iyi_sag_inds) > minpix:
            sagx_egri = np.int(np.mean(nonzerox[iyi_sag_inds]))
   

    sol_serit_inds = np.concatenate(sol_serit_inds)
    sag_serit_inds = np.concatenate(sag_serit_inds)
    
    solx = nonzerox[sol_serit_inds]
    soly = nonzeroy[sol_serit_inds]
    sagx = nonzerox[sag_serit_inds]
    sagy = nonzeroy[sag_serit_inds]

    # Eğrilere uyması için 2. derece polinom polyfit eğitildiği yer
    sol_fit = np.polyfit(soly, solx, 2)
    sag_fit = np.polyfit(sagy, sagx, 2)

    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    sol_fitx = sol_fit[0] * ploty**2 + sol_fit[1] * ploty + sol_fit[2]
    sag_fitx = sag_fit[0] * ploty**2 + sag_fit[1] * ploty + sag_fit[2]

    ltx = np.trunc(sol_fitx)
    rtx = np.trunc(sag_fitx)
    # plt.plot(sag_fitx)
    # plt.show()
    #polyfit  için şerit cizgilerin renklerinin verildiği yer
    out_img[nonzeroy[sol_serit_inds], nonzerox[sol_serit_inds]] = [255, 0, 0]
    out_img[nonzeroy[sag_serit_inds], nonzerox[sag_serit_inds]] = [0, 0, 255]

    global im_out_img
    im_out_img = out_img
    plt.imshow(out_img)
    plt.plot(sol_fitx,  ploty, color = 'yellow')
    plt.plot(sag_fitx, ploty, color = 'yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    return ploty, sol_fit, sag_fit, ltx, rtx

################################################################################



################################################################################
#### Başladı Yer-EĞRİLERİ TESPİT ETMEK İÇİN GENEL ARAMA YÖNTEMİNİ UYGULAYANDIĞI Kısım ######################
def Genel_Arama(binary_warped, sol_fit, sag_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    sol_serit_inds = ((nonzerox > (sol_fit[0]*(nonzeroy**2) + sol_fit[1]*nonzeroy +
    sol_fit[2] - margin)) & (nonzerox < (sol_fit[0]*(nonzeroy**2) +
    sol_fit[1]*nonzeroy + sol_fit[2] + margin)))
    sag_serit_inds = ((nonzerox > (sag_fit[0]*(nonzeroy**2) + sag_fit[1]*nonzeroy +
    sag_fit[2] - margin)) & (nonzerox < (sag_fit[0]*(nonzeroy**2) +
    sag_fit[1]*nonzeroy + sag_fit[2] + margin)))
    solx = nonzerox[sol_serit_inds]
    soly = nonzeroy[sol_serit_inds]
    sagx = nonzerox[sag_serit_inds]
    sagy = nonzeroy[sag_serit_inds]
    sol_fit = np.polyfit(soly, solx, 2)
    sag_fit = np.polyfit(sagy, sagx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    #Çizim için x ve y değerleri oluşturun
    sol_fitx = sol_fit[0]*ploty**2 + sol_fit[1]*ploty + sol_fit[2]
    sag_fitx = sag_fit[0]*ploty**2 + sag_fit[1]*ploty + sag_fit[2]


    ## GÖRSELLEŞTİRME ###########################################################

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[sol_serit_inds], nonzerox[sol_serit_inds]] = [255, 0, 0]
    out_img[nonzeroy[sag_serit_inds], nonzerox[sag_serit_inds]] = [0, 0, 255]
    
    sol_serit_pencere1 = np.array([np.transpose(np.vstack([sol_fitx-margin, ploty]))])
    sol_serit_pencere2 = np.array([np.flipud(np.transpose(np.vstack([sol_fitx+margin,
                                  ploty])))])
    sol_serit_pts = np.hstack((sol_serit_pencere1,  sol_serit_pencere2))
    sag_serit_pencere1 = np.array([np.transpose(np.vstack([sag_fitx-margin, ploty]))])
    sag_serit_pencere2 = np.array([np.flipud(np.transpose(np.vstack([sag_fitx+margin, ploty])))])
    sag_serit_pts = np.hstack((sag_serit_pencere1, sag_serit_pencere2 ))
    
     #addWeighted için arka plan renki vermek
    cv2.fillPoly(window_img, np.int_([sol_serit_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([sag_serit_pts]), (0, 255,0))
    sonuc = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    global im_pillpoly
    im_pillpoly = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


    # plt.imshow(sonuc)
    plt.plot(sol_fitx,  ploty, color = 'yellow')
    plt.plot(sag_fitx, ploty, color = 'yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    ret = {}
    ret['leftx'] = solx
    ret['rightx'] = sagx
    ret['left_fitx'] = sol_fitx
    ret['right_fitx'] = sag_fitx
    ret['ploty'] = ploty
    # print(solx)
    return ret
#### Bittiği Yer-EĞRİLERİ TESPİT ETMEK İÇİN GENEL ARAMA YÖNTEMİNİ UYGULAYANDIĞI Kısım ########################
################################################################################



################################################################################
#### BAŞLADIĞI YER - EĞRİ YARIÇAPINI ÖLÇME İŞLEVİ ##################################
def Serit_Egriligi_Ayarlamak(ploty, solx, sagx):
    
    solx = solx[::-1]  #Y'de yukarıdan aşağıya eşleştirmek için ters çevir
    sagx = sagx[::-1]  
    # Görüntünün altına karşılık gelen maksimum y değerini seçilir
    y_degerlendirme = np.max(ploty)
    #world space de x, y'ye yeni polinomları yerleştirilir
    sol_egit_cr = np.polyfit(ploty*ym_per_pix, solx*xm_per_pix, 2)
    sag_egit_cr = np.polyfit(ploty*ym_per_pix, sagx*xm_per_pix, 2)
     
    # Yeni eğrilik yarıçapı hesaplanır
    sol_egrilik  = ((1 + (2*sol_egit_cr[0]*y_degerlendirme*ym_per_pix + sol_egit_cr[1])**2)**1.5) / np.absolute(2*sol_egit_cr[0])
    sag_egrilik = ((1 + (2*sag_egit_cr[0]*y_degerlendirme*ym_per_pix + sag_egit_cr[1])**2)**1.5) / np.absolute(2*sag_egit_cr[0])
    #Şimdi eğrilik yarıçapımız metre cinsindendir
    # print(sol_egrilik, 'm', sag_egrilik, 'm')
    # print(solx[0],solx[1],)
    # Sol veya sağ eğri olduğuna karar verilir.
    global kavis

    if solx[0] - solx[-1] > 60:
        egri_yonlendirme = 'Left Curve'
        kavis = cv2.imread('turn-left.png')
                
    elif sagx[-1] - sagx[0] > 60:
        egri_yonlendirme = 'Right Curve '
        kavis = cv2.imread('turn-right.png')
    else:
        egri_yonlendirme = 'Straight'
        kavis = cv2.imread('go.png')

    return (sol_egrilik + sag_egrilik) / 2.0, egri_yonlendirme
#### Bittiği YER - EĞRİ YARIÇAPINI ÖLÇME İŞLEVİ ####################################
################################################################################



################################################################################
#### Başladığı Yer - ALGILANAN ŞERİTLER ALANINI GÖRSEL OLARAK GÖSTERME FONKSİYONU #####################
def Serit_Cizmek(original_image, warped_image,  min_Pencere, draw_info):

    solx = draw_info['leftx']
    sagx = draw_info['rightx']
    sol_fitx= draw_info['left_fitx']
    sag_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_sol= np.array([np.transpose(np.vstack([sol_fitx, ploty]))])
    pts_sag = np.array([np.flipud(np.transpose(np.vstack([sag_fitx, ploty])))])
    pts = np.hstack((pts_sol, pts_sag))

    ortalama_x = np.mean((sol_fitx, sag_fitx), axis=0)
    pts_ortalama = np.array([np.flipud(np.transpose(np.vstack([ortalama_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([pts_ortalama]), (0, 255, 255))

    newwarp = cv2.warpPerspective(color_warp,  min_Pencere, (original_image.shape[1], original_image.shape[0]))
    sonuc = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    global im_newwarp
  
    im_newwarp = cv2.warpPerspective(color_warp,  min_Pencere, (original_image.shape[1], original_image.shape[0]))
    return pts_ortalama, sonuc
#### Bittiği Yer -  ALGILANAN ŞERİTLER ALANINI GÖRSEL OLARAK GÖSTERME FONKSİYONU #######################
################################################################################


#### Başladığı Yer - ŞERİT MERKEZİNDEN SAPMAYI HESAPLAMAK İÇİN FONKSİYON ##################
################################################################################
def SeritM_Sapma(ortalamaPts, inpFrame):

    # Metre cinsinden sapmanın hesaplanması
    mpts =ortalamaPts[-1][-1][-2].astype(int)
    #Merkezden ne kadar uzağa saptığı metre cinsinden gösterilmesi
    pixelSapma = inpFrame.shape[1] / 2 - abs(mpts)
    deviation = pixelSapma * xm_per_pix
    print(deviation)
    direction = "Left" if deviation < 0 else "Right"

    return deviation, direction
################################################################################
#### Bittiği Yer - ŞERİT MERKEZİNDEN wSAPMAYI HESAPLAMAK İÇİN FONKSİYON ####################



################################################################################
#### BAŞLADIĞI YER - FİNAL GÖRÜNTÜYE BİLGİ METNİ EKLEME FONKSİYONU ##########################
def YaziEklemek(img, radius, direction, deviation, devDirection):

    # EğriYarıçapı ve merkez konumunu resme ekleyin
    font = cv2.FONT_HERSHEY_TRIPLEX

    if (direction != 'Düz'):
        text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius) + 'm'
        text1 = 'Curve Direction: ' + (direction)

    else:
        text = 'Radius of Curvature: ' + 'N/A'
        text1 = 'Curve Direction: ' + (direction)
    cv2.rectangle(img, (800,175),(1250,270), (0,0,0), -1)
    cv2.putText(img, text , (800,200), font, 0.8, (0,100, 200), 1, cv2.LINE_AA)
    cv2.putText(img, text1, (800,230), font, 0.8, (0,100, 200), 1, cv2.LINE_AA)

    # Sapma
    deviation_text = 'Off Center: ' + str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
    cv2.putText(img, deviation_text, (800, 260), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,100, 200), 1, cv2.LINE_AA)

    # putText Detayı
    cv2.rectangle(img, (1,155),(790,180), (0,0,0), -1)
    cv2.putText(img, "-- warpPerspective", (1,170), font, 0.5, (0,100, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "-- polyfit", (200,170), font, 0.5, (0,100, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "-- addWeighted", (400,170), font, 0.5, (0,100, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "-- newwarp", (600,170), font, 0.5, (0,100, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "-- viraj", (800,170), font, 0.5, (0,100, 200), 1, cv2.LINE_AA)


    return img
####  BİTTİĞİ YER - FİNAL GÖRÜNTÜYE BİLGİ METNİ EKLEME FONKSİYONU ############################
################################################################################

################################################################################
######## SON - GÖRÜNTÜ İŞLEME YAPILACAK FONKSİYONLAR ###########################
################################################################################

################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
######## BAŞLANGIÇ - MAIN FUNCTION #################################################
################################################################################

# Giriş resmini okuyun
image = readVideo()

################################################################################
#### - GİRİŞ GÖRÜNTÜSÜNÜ OYNATMAK İÇİN DÖNGÜ ######################################
while True:

    grabbed, frame = image.read()
    if grabbed == True:

        #"PerspektifWarp ()" fonksiyonunu çağırarak perspektif çarpıtma uygulanır
        # Sonra onu (birdView) adlı değişkene atanır
        # Bu işlevi şunlarla sağlaınır:
        # Perspektif warping  uygulanacak bir görüntü (frame)
        birdView, birdViewL, birdViewR,  min_Pencere = PerspectiveWrp(frame)


        # "onIsleme ()" fonksiyonunu çağırarak görüntü işlemeyi uygulayın
        # Ardından ilgili değişkenleri (img, hls, gri tonlama, eşik, bulanıklık, canny) atanır
        # Bu işlevi şunlarla sağlanır:
        # 1- işlenecek perspektif zaten  bir warped image(birdView)
        img, hls, grayscale, thresh, blur, canny = onIsleme(birdView)
        imgL, hlsL, grayscaleL, threshL, blurL, cannyL = onIsleme(birdViewL)
        imgR, hlsR, grayscaleR, threshR, blurR, cannyR = onIsleme(birdViewR)


        # "Get_histogram ()" fonksiyonu çağırarak histogramı çizin ve görüntüleyin
        # Bu fonksiyonu şunlarla sağlanır:
        # 1- (thresh) üzerinde ki  histogramı hesaplamak için bir görüntü.
        hist, leftBase, rightBase = Histogram(thresh)
        # print(rightBase - leftBase)
        # plt.plot(hist)
        # plt.show()


        ploty, sol_fit, sag_fit, sol_fitx, sag_fitx = Slide_Window_Search(thresh, hist)
        plt.plot(sol_fit)
        # plt.show()


        draw_info = Genel_Arama(thresh, sol_fit, sag_fit)  
        # plt.show()

        curveRad, curveDir = Serit_Egriligi_Ayarlamak(ploty, sol_fitx, sag_fitx)


        # Tespit edilen şeritlerin alanını yeşil ile doldurmak
        ortalamaPts, sonuc =Serit_Cizmek(frame, thresh,  min_Pencere, draw_info)


        deviation, directionDev = SeritM_Sapma(ortalamaPts, frame)
        
                #GÜNCELLEME EXCELL ----------------------------------------------------   
#
#         derece = deviation
#         yon = curveDir
#         
#         planSheet.write( sayac , 0 , derece )
#         planSheet.write( sayac , 1 , yon )
#         sayac += 1
#
        #GÜNCELLEME EXCELL ----------------------------------------------------


        # Son resmimize metin eklemek
        finalImg =  YaziEklemek(sonuc, curveRad, curveDir, deviation, directionDev)



        main = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        rows1,cols1,channels1 = frame.shape
        main[0:rows1, 0:cols1] = finalImg

        merge_o = cv2.resize(im_out_img, (190, 150)) # Resize image  
        #cv2.imshow('pillpoly',merge)

        merge_n = cv2.resize(im_newwarp, (190, 150)) # Resize image  
        #cv2.imshow('pillpoly',merge)

        merge_b = cv2.resize(kusbakisim, (190, 150)) # Resize image  
        #cv2.imshow('pillpoly',merge)

        merge_p = cv2.resize(im_pillpoly, (190, 150)) # Resize image  
        # cv2.imshow('pillpoly',merge)
        rows5,cols5,channels5 = kavis.shape
        main[0:rows5, 0+800:cols5+800] = kavis

        rows,cols,channels = merge_b.shape
        main[0:rows, 0:cols] = merge_b

        rows4,cols4,channels4 = merge_o.shape
        main[0:rows4, 0+200:cols4+200] = merge_o

        rows2,cols2,channels2 = merge_p.shape
        main[0:rows2, 0+400:cols2+400] = merge_p

        rows3,cols3,channels3 = merge_n.shape
        main[0:rows3, 0+600:cols3+600] = merge_n
        # Displaying final image
        cv2.imshow("Final", main)


        # Oynatmayı durdurmak enter tuşuna  veya oynatmayı bitirmek için q tuşuna basılması beklenir.
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    else:
        break

#### BİTTİŞ - GİRİŞ RESMİNİ OYNATMAK İÇİN DÖNGÜ ########################################
################################################################################

# Cleanup
image.release()
cv2.destroyAllWindows()

#planWorkbook.close() #excell ile ilgili


################################################################################
######## BİTTİŞ - MAIN FUNCTION ###################################################
################################################################################


































##
