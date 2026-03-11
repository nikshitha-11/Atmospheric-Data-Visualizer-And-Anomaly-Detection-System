from reportlab.pdfgen import canvas

def generate_report(city,temp,pm):

    file="environment_report.pdf"

    c=canvas.Canvas(file)

    c.drawString(100,800,"Atmospheric Environmental Report")
    c.drawString(100,760,f"City: {city}")
    c.drawString(100,740,f"Temperature: {temp}")
    c.drawString(100,720,f"PM2.5: {pm}")

    c.save()

    return file