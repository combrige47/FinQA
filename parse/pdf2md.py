from pathlib import Path

import opendataloader_pdf

class Pdf2Md:
    def __init__(self):
        backend:str = 'openpdf'
    def parse(self,input_path:str,output_path:str):
        opendataloader_pdf.convert(
            input_path=[input_path,],
            output_dir=output_path,
            format="markdown"
        )
    def batch_parse(self,input_path:str,output_path:str):
        input_path_obj = Path(input_path)
        opendataloader_pdf.convert(
            input_path=[str(input_path_obj),],
            output_dir=output_path,
            format="markdown"
        )

if __name__ == '__main__':
    pdf2md = Pdf2Md()
    pdf2md.batch_parse("../安道麦A","test_output")