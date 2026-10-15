# Metode Numerik Menggunakan Python — NM26

Repositori pendamping buku ajar S1 Teknik Informatika.

## Sumber kanonis dan kebijakan

Notebook utama setiap bab adalah `BabNN_kode_sumber_R53.ipynb`, diekstrak dari Word Master Revision 53. Kode sumber tidak ditulis ulang untuk membuatnya tampak berjalan. Fragmen yang membutuhkan konteks dari manuskrip dipertahankan sebagai teks berlabel, bukan dieksekusi. Pseudocode dan keluaran juga tetap dipisahkan dari kode Python.

Bab 9 memakai skrip `Bab09_Studi_Kasus/bab9_studi_kasus_metode_numerik.py`.

## Instalasi

```bash
python -m pip install -r requirements.txt
```

## Menjalankan studi kasus Bab 9

```bash
cd Bab09_Studi_Kasus
python bab9_studi_kasus_metode_numerik.py
```

## QA

Baca `QA_PENCOCOKAN_WORD_MASTER_R53.md`, `QA_RELEASE_SCREENING.json`, dan laporan per bab. Notebook sumber telah diuji dengan mengeksekusi sel kode secara berurutan dalam namespace Python terisolasi untuk setiap bab. Sel yang membutuhkan konteks yang tidak tersedia diklasifikasikan sebagai teks sumber. Pemeriksaan ini tidak menggantikan verifikasi independen seluruh hasil matematis.

## Status

**Source-matched GitHub release candidate.** Sumber yang dipakai tercatat di `SOURCE_PROVENANCE.txt`. Belum ada klaim bahwa repositori remote telah diperbarui.

Target repository: https://github.com/PedroJoTe/NM26
