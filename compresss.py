import aspose.words as aw

doc = aw.Document("testDoc.docx")

# DocSaveOptions, FixedPageSaveOptions, HtmlFixedSaveOptions, .......
save_options = aw.saving.OoxmlSaveOptions(aw.SaveFormat.DOCX)
# CompressionLevel.MAXIMUM, CompressionLevel.NORMAL, CompressionLevel.FAST, CompressionLevel.SUPER_FAST
save_options.compression_level = aw.saving.CompressionLevel.NORMAL

doc.save("document_compression.docx", save_options)