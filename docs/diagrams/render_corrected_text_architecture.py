"""Render the corrected text diagrams from docs/architecture_corrected.md.

This is a manual code-audited diagram specification, not automatic graph tracing.
Run from any directory: python docs/diagrams/render_corrected_text_architecture.py
"""
import re
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.colors import HexColor

from render_architecture_pdf import fonts, ROOT


def main():
    fonts()
    source = ROOT / 'docs/architecture_corrected.md'
    content = source.read_text(encoding='utf-8')
    sections = re.findall(r'^## (.*?)\n\n(.*?)\n\n```text\n(.*?)\n```', content, re.M | re.S)
    assert len(sections) == 6
    target = ROOT / 'output/pdf/temporal_bora_corrected_text.pdf'
    target.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(target), pagesize=(840,1188), pageCompression=1)
    c.setTitle('Temporal BORA-Fuse - Corrected text architecture')
    c.setAuthor('Architecture audited against local code')
    for i,(title,subtitle,diagram) in enumerate(sections,1):
        c.setFillColor(HexColor('#536579'))
        c.setFont('Bold',10)
        c.drawString(38,1153,'DUAL-DECODER TEMPORAL BORA-FUSE / SƠ ĐỒ TEXT ĐÃ HIỆU CHỈNH')
        c.setFont('Bold',22)
        c.setFillColor(HexColor('#172B42'))
        assert pdfmetrics.stringWidth(title,'Bold',22)<764
        c.drawString(38,1120,title)
        c.setFont('Body',10.5)
        assert pdfmetrics.stringWidth(subtitle,'Body',10.5)<764
        c.drawString(38,1093,subtitle)
        c.setStrokeColor(HexColor('#9DAEBC'))
        c.line(38,1079,802,1079)
        lines=diagram.splitlines()
        size=10.3
        leading=14.1
        assert 1055-(len(lines)-1)*leading>90, (i,len(lines))
        c.setFont('Mono',size)
        for j,line in enumerate(lines):
            assert pdfmetrics.stringWidth(line,'Mono',size)<=764, (i,j,line)
            c.setFillColor(HexColor('#172B42'))
            c.drawString(38,1055-j*leading,line)
        c.setFont('Body',9)
        c.setFillColor(HexColor('#536579'))
        c.drawString(38,62,'Nguồn chi tiết và bảng các điểm sửa: docs/architecture_corrected.md')
        c.line(38,48,802,48)
        c.drawString(38,30,'B=batch; T=8; D=256; k=0,1,2. Tensor cùng tên được truyền tiếp giữa các trang.')
        c.drawRightString(802,30,f'{i:02d} / 06')
        c.showPage()
    c.save()
    text_path = target.with_suffix('.txt')
    text_path.write_text('\n\n'.join(title+'\n'+'='*len(title)+'\n'+diagram
                                    for title,_,diagram in sections),encoding='utf-8')
    print(target)
    print(text_path)


if __name__ == '__main__':
    main()
