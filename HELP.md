# Curvelens – Yardım

| | |
|---|---|
| **Program** | Curvelens |
| **Sürüm** | 1.1.0 |
| **Hazırlayan** | Efe Arda CENGİZ |
| **Tarih** | 2 Ekim 2026 |

Curvelens, PDF belgelerindeki ve grafik görsellerindeki eğrileri sayısal tablolara (lookup tablosu) dönüştüren bir
araçtır. Her değer, kaynağın çözünürlüğünden türetilen bir belirsizlikle (±) birlikte verilir. Yöntemin ayrıntıları
için `README.md` dosyasına bakın.

## 1. Programı başlatma

| Yöntem | Nasıl |
|---|---|
| **Hazır exe** (kurulum gerektirmez) | GitHub'daki *Releases* bölümünden `Curvelens.exe` dosyasını indirip çift tıklayın. İlk açılış birkaç saniye sürer. |
| **Kaynak koddan** (Windows) | `run.bat` dosyasına çift tıklayın. İlk çalıştırmada Python sanal ortamı ve kütüphaneler otomatik indirilir (internet gerekir), sonraki açılışlar doğrudan başlar. |
| **Elle** | `python -m venv .venv`, ardından `.venv\Scripts\python -m pip install -r requirements-run.txt`, ardından `.venv\Scripts\python app.py` |

Gereksinim: Windows 10/11 (64 bit). Kaynak koddan çalıştırmak için Python 3.11 veya daha yenisi.

## 2. Hızlı başlangıç

1. **Dosya açın:** `Ctrl+O` ya da dosyayı pencereye sürükleyip bırakın. Desteklenenler: PDF, PNG, JPG, BMP, TIFF, WebP.
2. **Galeriden grafiği seçin.** PDF'teki tüm grafikler otomatik bulunup önizlemeyle listelenir.
3. **Eğriyi seçin.** Eğri vurgulanır ve sağda lookup tablosu oluşur.
4. **Değer sorgulayın:** `X = …` yazarsanız Y değerini, `Y = …` yazarsanız X değerini belirsizliğiyle birlikte görürsünüz.
5. **Kaydedin:** tabloyu CSV ya da PDF olarak dışa aktarın veya yazdırın.

## 3. Grafik türüne göre çalışma

**Vektör grafikler** (PDF'e çizgi olarak çizilmiş): kalibrasyon gerekmez, değerler doğrudan PDF'in çizim verisinden okunur.

**Resim grafikler** (PDF'e görüntü olarak gömülü ya da bağımsız görsel dosyası): galeride
"görsel grafik · kalibrasyon gerekli" olarak görünür.

1. Grafik çerçevesi otomatik bulunamazsa **Grafik alanı…** ile eksenlerin sınırladığı dikdörtgeni çizin.
2. Her eksende iki işaret (tick) tıklayıp değerlerini girin. İkinci bir Y ekseni varsa iki işaret daha seçin.
   Birim parantez içinde yazılırsa (ör. `Gerilim (V)`) tablo başlığına aktarılır.
3. Eğriler renklerine göre otomatik ayrılır. Eksik eğri için **＋ Eğri ekle**, hatalı eğri için **Eğriyi sil**,
   yanlış eksene atanan eğri için **Eksen ⇄**, ad vermek için **Adı…** (`F2`) kullanın.

Kalibrasyon bir kez yapılır ve hatırlanır (aynı PDF'in aynı grafiği için bir daha sorulmaz). Güncellemek için
**Yeniden kalibre et**, silmek için `Araçlar → Bu grafiğin kayıtlı kalibrasyonunu sil`.

## 4. Elle düzeltme

Hatalı okunan bölgeler şu araçlarla düzeltilir: **Nokta ekle**, **Nokta sil**, **Kutuyla sil**, **Nokta taşı**.
**Özgün hale döndür** tüm düzeltmeleri geri alır. Düzeltilen eğriler listede "düzeltildi (N)" olarak işaretlenir ve
çıktılara not olarak yazılır.

## 5. Tablo karşılaştırma

`Dosya → Tablo içe aktar…` (`Ctrl+I`) ile CSV/TXT/TSV biçiminde bir tabloyu içe aktarıp bir grafik eğrisiyle
karşılaştırabilirsiniz. Noktalar grafik üzerine çizilir; ortalama fark, RMS ve en büyük fark hesaplanır. Eşleştirme
yapmazsanız tablodan yeni bir grafik çizilir. Sonuçlar PNG ve CSV olarak dışa aktarılır.

## 6. Kısayollar ve fare

| Kısayol | İşlev |
|---|---|
| `Ctrl+O` | Dosya aç |
| `Ctrl+I` | Tablo içe aktar |
| `Ctrl+Z` / `Ctrl+Y` | Geri al / yinele |
| `F2` | Eğriyi yeniden adlandır |
| Sol fare tuşu | İşaret seç / araç işlemi |
| Sağ ya da orta tuşla sürükleme | Görüntüyü kaydır |
| Fare tekerleği | Yakınlaştır / uzaklaştır |

## 7. Komut satırı seçenekleri

```
run.bat dosya.pdf                       belirtilen dosyayı açarak başlatır
python app.py --manual resim.png        elle sayısallaştırma aracını açar (gelişmiş)
python app.py --selftest dosya.pdf      kütüphane ve paket denetimini çalıştırır
python app.py --screenshot out.png dosya.pdf --page gallery|detail --chart N --curve M
```

## 8. Dosya konumları

- Kayıtlı kalibrasyonlar: `%APPDATA%\Curvelens\calibrations.json`. Konum `CURVELENS_DATA` ortam değişkeniyle
  değiştirilebilir.
- Program eskiden "Plot Digitizer" adını taşıyordu. Eski `%APPDATA%\PlotDigitizer` klasörü varsa kayıtlar oradan
  okunmaya devam eder.

## 9. Sık karşılaşılan durumlar

| Durum | Çözüm |
|---|---|
| Windows SmartScreen "Bilinmeyen yayıncı" uyarısı veriyor | Program imzasızdır. **Ek bilgi → Yine de çalıştır** seçin. |
| `Curvelens.exe` ilk açılışta geç açılıyor | Tek dosyalık exe her açılışta geçici klasöre açılır; birkaç saniye normaldir. |
| `run.bat` Python bulamıyor | Python 3.11+ kurun (`winget` varsa otomatik denenir) ve `run.bat` dosyasını yeniden çalıştırın. |
| `run.bat` kurulumda hata veriyor | İnternet bağlantısını kontrol edip yeniden çalıştırın. Sorun sürerse `.venv` klasörünü silip tekrar deneyin. |
| Tarama (resim) PDF'inde grafik bulunamıyor | Sayfayı görsel olarak kaydedip açın ya da `--manual` aracını kullanın. |
| Çerçevesiz vektör grafik bulunamıyor | Otomatik tespit çerçeve kenarlarına dayanır; grafiği görsel olarak açıp **Grafik alanı…** ile tanımlayın. |

Ayrıntılı sınırlamalar için `README.md` dosyasındaki "Sınırlamalar" bölümüne bakın.
