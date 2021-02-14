#include <Servo.h>
#include <SoftwareSerial.h>


SoftwareSerial esp(10,11);
Servo servo;

String okunan;
String yon;
String teker;
String aci;



void setup()  
{
  servo.attach(8);
  
  Serial.begin(115200); /* bilgisayarla haberleşmeyi başlatıyoruz */

  esp.begin(115200);


    esp.println("AT");
    
    while(!esp.find("OK"))
    {
      esp.println("AT");
      Serial.println("ESP BULUNAMADI");
      }


  
   esp.println("AT+CWMODE=1");
   while(!esp.find("OK"))
   {
    esp.println("AT+CWMODE=1");
    Serial.println("CWMODE AYARI");
    }

  


  esp.println("AT+CWJAP=\"gokov\",\"gokhangokov35\"");
  while(!esp.find("OK"));

   Serial.println("Ağa Bağlandı");

}

void loop()
{   
 
 

  
 // delay(100);
   esp.println("AT+CIPSTART=\"TCP\",\"192.168.0.12\",5000");
   if(esp.find("Error"))
   {
    Serial.println("cipstart hatası");
    }
   // delay(100);

   String veri = "GET /\r\n\r\n";
   
  
   esp.print("AT+CIPSEND=");

   esp.println(veri.length() + 2);

   delay(100);




    esp.print(veri);
    esp.print("\r\n\r\n");
    Serial.println("veri gönderildi");
    
    

    while(esp.read()!= 95);//95= _ 

    okunan = esp.readString();

    yon = okunan.substring(0,1);

    teker = okunan.substring(1,2);

    aci = okunan.substring(2,4);
    
    
  
    Serial.println("yon:"+ yon );

    Serial.println("teker:"+ teker );

    Serial.println("aci:"+ aci );
	
	/*  İnte çevrilip sonra tekrar texte dönüştürülecek 
	
	if(yon == "0") //Sol
	{		
		Servo.write( 90 - aci );
	}
    else if(yon == "1") //Düz
	{		
		Servo.write( 90 );
	}
	else if(yon == "2") //Sağ
	{		
		Servo.write( 90 + aci );
	}
   
	*/
    

      esp.println("AT+CIPCLOSE");
      Serial.println("bağlantı kapandı");

      delay(100);
      

    ////////////////////////////////////////////////////////////////////////


}
