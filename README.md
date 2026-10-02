# Curvelens

**PDF belgelerindeki ve grafik görsellerindeki eğrilerin sayısal değerlere dönüştürülmesi için bir yazılım aracı**

## Özet

Teknik belgelerde (ör. bileşen veri sayfaları) sunulan karakteristik eğriler çoğunlukla yalnızca grafik biçiminde
yayımlanır; bu eğrilerin sayısal analizde, benzetimde ya da gömülü yazılımlarda kullanılabilmesi için tablo (lookup
tablosu) biçimine dönüştürülmesi gerekir. Curvelens, bu dönüşümü iki farklı veri kaynağı için gerçekleştirir:
(i) PDF sayfasına vektör olarak çizilmiş grafiklerde eğri değerleri, piksel ölçümüne başvurulmadan doğrudan PDF'in vektör
geometrisinden elde edilir; (ii) PDF'e raster görüntü olarak gömülmüş grafiklerde ve bağımsız görsel dosyalarında (PNG,
JPG, BMP, TIFF, WebP) değerler, kullanıcı tarafından yapılan eksen kalibrasyonu ve renk tabanlı eğri ayrıştırması ile
okunur. Her iki yöntemde de her değer, kaynağın çözünürlüğünden türetilen bir belirsizlikle (±) birlikte raporlanır.
Elde edilen tablolar üzerinde X→Y ve Y→X sorgusu, elle düzeltme, harici tablolarla karşılaştırma ve CSV/PDF çıktısı
alınabilir.

**Bağımlılıklar:** Python 3.11+, PySide6, NumPy, SciPy, pandas, pypdfium2, OpenCV (raster grafikler ve elle
sayısallaştırma aracı için).

## 1. Yöntem

### 1.1 Vektör grafiklerden değer çıkarımı

1. **Grafik bölgesinin tespiti.** Grafik çerçevesi, dört ince kenar çizgisinden oluşan dikdörtgen olarak aranır; başlık,
   eksen adları ve tick etiketleri PDF'in metin katmanından çözümlenir.
2. **Eksen kalibrasyonu.** Sayfa koordinatları ile veri değerleri arasındaki dönüşüm, yalnızca iki referans noktası
   yerine **tüm tick etiketleri** kullanılarak en küçük kareler yöntemiyle kestirilir (doğrusal ya da log₁₀ ölçek).
3. **Eğri ayrıştırması.** Çizim yolları geometrik özelliklerine göre sınıflandırılır: uç uca bitişen parçalar sürekli
   eğri, düzenli aralıklı kısa parçalar kesikli eğri, küçük kapalı şekiller işaretçi serisi olarak yorumlanır. Izgara,
   çerçeve, tick ve legend çizgileri ayıklanır. Vektör verisi örtülen bölgelerde de mevcut olduğundan, başka bir eğrinin
   altında kalan kesimler de eksiksiz okunur.
4. **Anlamlandırma.** Her eğri; legend işareti ve bitişiğindeki metinle, legend bulunmadığında eksen rengi (ör. mavi eksen
   → mavi eğri) ya da çizgi türüyle eşleştirilir. Çok Y eksenli grafiklerde eğri, rengi eşleşen eksenin ölçeğinde okunur.

### 1.2 Raster grafiklerden değer çıkarımı

Görüntü olarak gömülmüş grafikler galeride "görsel grafik · kalibrasyon gerekli" olarak listelenir (ad, görüntünün
üzerindeki PDF metninden alınır). Bağımsız görsel dosyaları aynı iş akışıyla tek grafik olarak açılır.

1. **Grafik alanı.** Çerçeve otomatik olarak aranır; bulunamadığında (çerçevesiz grafik ya da yüksek gürültü) eksenlerin
   sınırladığı dikdörtgen kullanıcı tarafından **Grafik alanı…** ile tanımlanır.
2. **Kalibrasyon.** Kullanıcı X ve Y eksenlerinin her birinde iki tick işaretine tıklar; tıklama konumu en yakın tick
   çizgisine ya da ızgara çizgisine hizalanır, çerçeve kenarları da aday olarak önerilir. İkinci bir Y ekseni varsa iki
   işaret daha seçilir. Sol fare tuşu işaret seçer; sağ (ya da orta) tuşla sürüklemek görüntüyü kaydırır,
   tekerlek yakınlaştırır (bu davranış tıklamalı tüm araçlar için geçerlidir). İşaretlerin sayısal değerleri ve eksen adları elle girilir (birim parantez içinde yazıldığında,
   ör. `Gerilim (V)`, tablo başlığına aktarılır); logaritmik eksenler ayrıca belirtilir. Metin tanıma (OCR) kullanılmaz.
3. **Eğri ayrıştırması.** Eğriler renk bileşenlerine göre otomatik olarak ayrılır; aynı renkteki iki eğri (ör. gerilim ve
   sıcaklık) ayrı ayrı bulunur ve sağ eksen renginde başlığı olan eğriler sağ eksene atanır. Legend görüntü içinde
   olduğundan eğriler `Eğri N (renk)` olarak adlandırılır ve **Adı…** (`F2`) ile yeniden adlandırılabilir. Eksik ya da
   siyah/gri eğriler **＋ Eğri ekle** ile (kesikli çizgiler için "Kesikli" seçeneğiyle) tohum noktasından izlenir; hatalı
   eğriler **Eğriyi sil** ile kaldırılır, yanlış eksene atananlar **Eksen ⇄** ile düzeltilir.
4. **Kalibrasyonun kalıcılığı.** Tamamlanan kalibrasyon, PDF'in dosya içeriğinden hesaplanan özet değeriyle
   ilişkilendirilerek saklanır; bu sayede dosyanın taşınması ya da yeniden adlandırılması kaydı etkilemez, içeriği
   değişmiş bir dosya ise yeniden kalibrasyon gerektirir. Kayıt yalnızca kalibrasyonu (ve varsa elle tanımlanan grafik
   alanını) içerir; eğri adları ve elle düzeltmeler oturumla sınırlıdır. Kayıt dosyası
   `%APPDATA%\Curvelens\calibrations.json` konumundadır ve `CURVELENS_DATA` ortam değişkeniyle değiştirilebilir
   (program eskiden "Plot Digitizer" adını taşıyordu; eski `%APPDATA%\PlotDigitizer` klasörü varsa o kullanılmaya devam eder).
   Kalibrasyon **Yeniden kalibre et** ile güncellenir, `Araçlar → Bu grafiğin kayıtlı kalibrasyonunu sil` ile silinir.
   Vektör grafikler kalibrasyon gerektirmez.

### 1.3 Tablo üretimi, interpolasyon ve belirsizlik

* **Örnekleme.** X adımı varsayılan olarak yaklaşık 50 satır verecek biçimde yuvarlak bir değere ayarlanır; kullanıcı
  tarafından değiştirilebilir (`0`: eğrinin özgün noktaları). Grafikte işaretlenen noktalar tablonun satırlarıyla
  birebir örtüşür.
* **İnterpolasyon.** Ardışık noktalar arasında doğrusal interpolasyon uygulanır (kesikli çizgilerde boşluklar dahil).
  Eğrinin tanım aralığı dışında ekstrapolasyon yapılmaz.
* **Nokta belirsizliği.** Vektör grafiklerde belirsizlik, PDF'in koordinat çözünürlüğüdür; raster grafiklerde ise
  görüntünün piksel çözünürlüğüdür. Elle eklenen noktaların belirsizliği tıklamanın piksel çözünürlüğüne eşittir.
* **Sorgu belirsizliği.** Değer sorgusunda nokta belirsizlikleri (σₓ, σᵧ), eğrinin yerel eğimi *m* ile birleştirilir;
  *m*, sorgu noktası çevresindeki komşu noktalara en küçük kareler doğrusu uydurularak kestirilir:

  $$\sigma_Y = \sqrt{\sigma_y^2 + (m\,\sigma_x)^2}, \qquad \sigma_X = \sqrt{\sigma_x^2 + (\sigma_y / |m|)^2}$$

  Dolayısıyla eğimin yüksek olduğu bölgelerde Y belirsizliği, eğrinin yataya yakın olduğu bölgelerde X belirsizliği
  büyür; eğimin sıfıra yaklaştığı durumlarda X değeri belirsiz (∞) olarak raporlanır. Eğri aynı Y değerini birden fazla
  kez aldığında tüm çözümler listelenir.

## 2. Doğrulama

* **Birim ve entegrasyon testleri**, matplotlib ile üretilmiş ve doğru değerleri önceden bilinen PDF grafikleri üzerinde
  çalıştırılır (`tests/`).
* **Vektör çıkarımı.** Örnek veri sayfasında koordinat çözünürlüğü ±0,06 pt olup bu değer yaklaşık ±1,2 mAh ve
  ±0,0026 V belirsizliğe karşılık gelir.
* **Raster çıkarımı.** Beş eğri içeren ve 600 DPI PNG olarak dışa aktarılmış bir deneme grafiğinde, raster yöntemle
  okunan değerlerin aynı grafiğin vektör değerlerinden sapması ortalama +0,003 V, en fazla 0,0044 V olarak ölçülmüştür;
  bu sapma bir pikselin altındadır. JPEG sıkıştırmasının çizgi kenarlarında yol açtığı bulanıklaşma bu sınırı aşmamıştır.
  Görüntü olarak gömülü grafik içeren bir veri sayfasında tipik belirsizlik ≈ ±2 mAh ve ±0,004 V düzeyindedir.
* **Referans değerlerle tutarlılık.** Şarj eğrisinden okunan değerler (1400 mA şarj akımı, 4,2 V şarj sonu gerilimi,
  140 mA kesme akımı, ≈2830 mAh) veri sayfasında beyan edilen değerlerle uyumludur.

## 3. Kullanım

1. **Dosya açma** (`Ctrl+O` ya da sürükle-bırak). PDF'in tüm sayfaları grafik için taranır (≈1 s/sayfa); görsel dosyaları
   tek grafik olarak açılır.
2. **Galeri.** Bulunan grafikler önizleme ve adlarıyla listelenir.
3. **Detay görünümü.** Grafik, eğri listesi (legend metni, ilişkili eksen, çizgi türü) ve grafik notları birlikte
   gösterilir. Seçilen eğri vurgulanır ve lookup tablosu oluşturulur; imleç konumundaki X, Y değerleri ile seçili eğrinin
   o X'teki değeri durum çubuğunda görüntülenir.
4. **Değer sorgusu.** `X = …` girdisi eğrinin Y değerini, `Y = …` girdisi X değer(ler)ini belirsizliğiyle birlikte
   döndürür; sonuç grafikte işaretlenir. Ondalık ayırıcı olarak virgül ya da nokta kabul edilir.
5. **Elle düzeltme.** Hatalı okunan bölgeler **Nokta ekle**, **Nokta sil**, **Kutuyla sil** ve **Nokta taşı** araçlarıyla
   düzeltilir (`Ctrl+Z` / `Ctrl+Y`; **Özgün hale döndür** tüm düzeltmeleri geri alır). Düzeltilmiş eğriler listede
   "düzeltildi (N)" olarak işaretlenir; tüm çıktılar düzeltilmiş veriyi kullanır ve bu durumu not olarak belirtir.
6. **Çıktı.** Tablo CSV ya da PDF olarak kaydedilir veya sistem yazdırma penceresi üzerinden yazdırılır.

### 3.1 Harici tablolarla karşılaştırma

**Tablo karşılaştırma** sekmesi (`Dosya → Tablo içe aktar…`, `Ctrl+I`), mevcut bir lookup tablosunun bir grafik eğrisiyle
nicel olarak karşılaştırılmasını sağlar. CSV/TXT/TSV biçimleri; `;` ya da sekme ayraçlı dosyalar, virgül ya da nokta
ondalık ayırıcısı, başlıklı ya da başlıksız ve çok sütunlu tablolar desteklenir.

* **Eşleştirmeli karşılaştırma.** Tablo bir grafik ve eğriyle eşleştirildiğinde noktalar grafik üzerine çizilir ve
  şu istatistikler hesaplanır: eğrinin X aralığındaki nokta sayısı; tablo − eğri farkının ortalaması, ortalama mutlak
  değeri, RMS değeri, en büyük değeri ve bağıl yüzdesi; eğrinin belirsizliği. Satır bazında eğri değeri, fark ve bağıl
  fark ayrıca listelenir. Birim uyuşmazlıkları (ör. mAh ↔ Ah) uyarı olarak bildirilir.
* **Eşleştirmesiz görselleştirme.** Eşleştirme yapılmadığında tablodan eksenleri, ızgarası ve legend'ı olan yeni bir
  grafik çizilir.
* Grafik alanı dışında kalan satırlar sayılarak raporlanır. Eşleşen eğri düzeltildiğinde karşılaştırma otomatik olarak
  yenilenir. Sonuçlar PNG (grafik) ve CSV (karşılaştırma) olarak dışa aktarılabilir.

## 4. Sınırlamalar

* Otomatik tespit; vektör grafikleri, çerçeveli bir grafik içeren gömülü görüntüleri ve görsel dosyalarını kapsar.
  Sayfanın tamamının tek bir görüntü olduğu taranmış PDF'lerde sayfa görsel olarak kaydedilip açılmalı ya da
  `Araçlar → Görselden elle sayısallaştır (gelişmiş)` aracı (`python app.py --manual resim.png`) kullanılmalıdır.
* Grafik tespiti çerçeve kenarlarına dayandığından çerçevesiz vektör grafikler otomatik olarak bulunamaz.
* Log eksenlerde `10^n` biçimli (mathtext) tick etiketleri çözümlenemez; düz sayısal etiketler (1, 10, 100) desteklenir.
* Aynı renk ve çizgi türüne sahip iki ayrı eğri tek eğri olarak birleştirilir.
* Raster grafiklerde: eksen değerleri elle girilir (OCR yoktur); en fazla iki Y ekseni desteklenir; kesikli, siyah ya da
  gri eğrilerle çakışan eğriler için elle ekleme gerekebilir; başka bir eğrinin örttüğü kesimler görüntüde bulunmadığından
  okunamaz; kalın çerçeve nedeniyle eğrilerin eksenle kesiştiği ilk birkaç piksel okunamayabilir.
* Doğruluk, raster grafiklerde görüntünün piksel çözünürlüğüyle, vektör grafiklerde PDF'in koordinat hassasiyetiyle
  sınırlıdır.

## 5. Kurulum, çalıştırma ve test

**Kolay yol (Windows):** `run.bat` dosyasına çift tıklayın. İlk çalıştırmada gerekli kütüphaneler `.venv` içine otomatik
indirilir (internet gerekir; Python 3.11+ yoksa `winget` ile kurulması denenir), sonraki çalıştırmalarda program doğrudan açılır.

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py                                  # grafik arayüz
.\.venv\Scripts\python app.py samples\INR18650P28A-V1-80093.pdf
.\.venv\Scripts\python -m pytest                               # testler
.\.venv\Scripts\python app.py --selftest samples\INR18650P28A-V1-80093.pdf   # kütüphane/paket denetimi
```

Tek dosyalık çalıştırılabilir: `python build_exe.py` → `dist\Curvelens.exe` (ilk açılış birkaç saniye sürer).
`--screenshot out.png dosya.pdf --page gallery|detail --chart N --curve M` pencere görüntüsünü dosyaya kaydeder.

## 6. Yazılım mimarisi

```
app.py                     giriş noktası (+ --selftest / --screenshot / --manual)
core/
  vector_charts.py         PDF vektör verisinden grafik, eğri, anlam, eksen uyumu, notlar
  pdf_source.py            PDF sayfa/bölge render, grafik çerçevesi, başlık, eksen/tick bulma
  lookup.py                X→Y tablosu, interpolasyon, ondalık basamak, HTML raporu, CSV
  query.py                 X→Y ve Y→X değer sorgusu (interpolasyon, çoklu çözüm, belirsizlik)
  curve_edit.py            elle düzeltme: nokta ekle/sil/taşı, geri al/yinele (GUI'siz)
  calibration_store.py     raster grafik kalibrasyonlarının içerik özetine göre saklanması (JSON)
  raster_charts.py         gömülü görüntü grafikleri: tespit, tick bulma, kalibrasyon, otomatik eğri okuma
  table_import.py          lookup tablosu okuma (CSV/TXT, ayraç/ondalık tahmini) + eğriyle karşılaştırma
  postprocess.py, calibration.py, models.py
  extraction.py, project.py, plotarea.py, export.py, imageio.py    görselden elle sayısallaştırma (eski araç)
ui/
  lookup_window.py         ana pencere: karşılama → galeri → detay (düzeltme, raster kalibrasyonu)
  query_panel.py           "Değer sorgula" paneli
  compare_tab.py           "Tablo karşılaştırma" sekmesi: içe aktarma, eşleştirme, grafik üzerinde gösterim
  plot_canvas.py           tablodan yeni grafik çizimi (eşleştirme yoksa)
  calibration_dialog.py    raster kalibrasyonda seçilen işaretlerin değer girişi
  printing.py              tablonun PDF'e dönüştürülmesi / yazdırılması (QTextDocument + QPrinter)
  image_view.py, table_model.py, appicon.py
  main_window.py, pdf_dialog.py                                    elle sayısallaştırma arayüzü (eski araç)
tests/                     birim, entegrasyon ve arayüz testleri; doğru değerleri bilinen matplotlib PDF'leri
samples/                   INR18650P28A-V1-80093.pdf (örnek veri sayfası), discharge.png
```

**Koordinat sistemi:** Tüm iç koordinatlar PDF noktası (pt) birimindedir; orijin sayfanın sol üst köşesidir ve y ekseni
aşağı doğru artar.
