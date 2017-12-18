#include "mainwidget.h"
#include "optionswidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QPushButton>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QDebug>
#include <QProgressBar>
#include <QMessageBox>
#include <QLayoutItem>
#include <QLabel>
#include <math.h>

void MainWidget::ZoomIn(const QPoint &p)  {
    double x = xOrigin - (2 / zoomFactor) * (1 - 2 * ((double) p.x() / imageWidth));
    double y = yOrigin - (2 / zoomFactor) * (1 - 2 * (((double) p.y() + (imageWidth - imageHeight) / 2) / imageWidth));
    xOrigin = x - ( x - xOrigin ) / zoomStep;
    yOrigin = y - ( y - yOrigin ) / zoomStep;
    zoomFactor *= zoomStep;
    BuildMandelbrot();
}

void MainWidget::keyPressEvent(QKeyEvent *ev)
{

        if (ev -> key() == Qt::Key_Left)
        {
            double step = 1 / zoomFactor;
            if ( ev -> modifiers() & Qt::ShiftModifier )
                step /= 10;
            xOrigin -= step;
            BuildMandelbrot();
        }
        if(ev -> key() == Qt::Key_Right)
        {
            double step = 1 / zoomFactor;
            if (ev -> modifiers() & Qt::ShiftModifier)
                step /= 10;
            xOrigin += step;
            BuildMandelbrot();
        }
        if( ev -> key() == Qt::Key_Up)
        {
            double step = 1 / zoomFactor;
            if(ev -> modifiers() & Qt::ShiftModifier)
                step /= 10;
            yOrigin -= step;
            BuildMandelbrot();
        }
        if(ev -> key() == Qt::Key_Down)
        {
            double step = 1 / zoomFactor;
            if(ev -> modifiers() & Qt::ShiftModifier)
                step /= 10;
            yOrigin += step;
            BuildMandelbrot();
        }
        if(ev -> key() == Qt::Key_Minus)
        {
            if(ev -> modifiers() & Qt::ControlModifier)
            {
                if(maxIterations > 100)
                    maxIterations -= 100;
            }
            else
            {
                if(ev -> modifiers() & Qt::ShiftModifier)
                {
                    if (maxIterations > 1)
                        maxIterations --;
                }
                else
                {
                    if (maxIterations > 10)
                        maxIterations -= 10;
                }
            }
            BuildMandelbrot();
        }

        if(ev -> key() == Qt::Key_Plus)
        {
            if(ev -> modifiers() & Qt::ControlModifier)
                 maxIterations += 100;
            else
            {
                if(ev -> modifiers() & Qt::ShiftModifier)
                    maxIterations ++;
                else
                    maxIterations += 10;
            }
            BuildMandelbrot();
        }

}

bool MainWidget::eventFilter(QObject *obj, QEvent *event)
{
    // imgLabel -> setFocus(); //TODO: fix crash here

    if(obj == this -> imgLabel)
    {
       if(event -> type() == QEvent::MouseButtonRelease)
        {
            if(isEnabled())
            {
                QMouseEvent* ev = static_cast<QMouseEvent*>(event);
                if(ev -> button() == Qt::LeftButton)
                {
                    ZoomIn(ev -> pos());
                    return true;
                }
                if(ev -> button() == Qt::RightButton)
                {
                    ZoomOut();
                    return true;
                }
            }
        }
    }

    return false;
}

void MainWidget::UpdatePixmap()
{
    imgLabel -> setPixmap(QPixmap::fromImage(*image));
}

void MainWidget::UpdateInfoLabel()
{
    QString infoString;
    infoString.append("Zoom: ");
    infoString.append(QString::number(zoomFactor));
    infoString.append(". ");
    infoString.append(QString::number(maxIterations));
    infoString.append(" Iterations.");
    zoomInfoLabel -> setText(infoString);

    infoString.clear();
    infoString.append("X = ");
    infoString.append(QString::number(xOrigin));
    infoString.append(". Y = ");
    infoString.append(QString::number(-yOrigin));
    infoString.append(".");
    posInfoLabel -> setText(infoString);
}

void MainWidget::BuildUi()
{
    mainVbLay       = new QVBoxLayout(this);
    zoomInfoLabel   = new QLabel(this);
    posInfoLabel    = new QLabel(this);
    buttonsHbLay    = new QHBoxLayout();
    pushReset       = new QPushButton(this);
    pushOptions     = new QPushButton(this);
    imgLabel        = new QLabel(this);
    image           = new QImage(imageWidth, imageHeight, QImage::Format_RGB32);

    imgLabel    -> setPixmap(QPixmap::fromImage(*image));
    pushReset   -> setText("Reset");
    pushReset   -> setFixedWidth(100);
    pushOptions -> setText("Options and Instructions");
    
    buttonsHbLay -> addStretch();
    buttonsHbLay -> addWidget(pushOptions);
    buttonsHbLay -> addWidget(pushReset);
    buttonsHbLay -> addStretch();

    mainVbLay -> addWidget(zoomInfoLabel, 0, Qt::AlignCenter);
    mainVbLay -> addWidget(posInfoLabel, 0, Qt::AlignCenter);
    mainVbLay -> addLayout(buttonsHbLay);
    mainVbLay -> addWidget(imgLabel, 0, Qt::AlignCenter);
    mainVbLay -> addStretch();
    mainVbLay -> setSizeConstraint(QLayout::SetFixedSize);

    setLayout(mainVbLay);

    setWindowFlags(windowFlags() ^ Qt::WindowMaximizeButtonHint);
}

void MainWidget::OpenOptions()
{
    OptionsWidget* optionsWindow;
    optionsWindow = new OptionsWidget(this, showInfo, isNormalized, saveImg);
    optionsWindow -> show();
    
    QPoint position = pos();
    position.setX(position.x() + width()  / 2 - ( optionsWindow -> width()  ) / 2 );
    position.setY(position.y() + height() / 2 - ( optionsWindow -> height() ) / 2 );

    optionsWindow -> move( position );

    setEnabled( false );
}

extern "C"
int* iterateGPU(int w, int h, int maxIterations, double xOrigin, double yOrigin, double zoomFactor);

void MainWidget::BuildMandelbrot()
{
    auto v = iterateGPU(imageWidth, imageHeight, maxIterations, xOrigin, yOrigin, zoomFactor);
    for(int p = 0; p < imageWidth * imageHeight; ++ p)
    {
        // std::cout << "i = " << p / imageWidth << " " << p % imageWidth << "\n";
        image -> setPixel(p % imageWidth, p / imageWidth, qRgb(v[p], v[p], v[p]));
    }
    delete[] v;

    if(isNormalized)
        Normalize();
    UpdatePixmap();
    if (showInfo)
        UpdateInfoLabel();
    if(saveImg)
    {
        QPixmap pixmap = QPixmap::fromImage(*image);
        pixmap.save("image.jpg", 0, 90);
    }
}

void MainWidget::ZoomOut()
{
    zoomFactor /= zoomStep;
    BuildMandelbrot();
}

void MainWidget::Reset()
{
    xOrigin = yOrigin = 0;
    zoomFactor = 1;
    maxIterations = 50;
    BuildMandelbrot();
}

void MainWidget::ToggleInfo()
{
    showInfo = !showInfo;
    if(showInfo)
    {
        zoomInfoLabel   = new QLabel(this);
        posInfoLabel  = new QLabel(this);
        mainVbLay -> insertWidget(0, zoomInfoLabel,  0, Qt::AlignCenter);
        mainVbLay -> insertWidget(1, posInfoLabel, 0, Qt::AlignCenter);
        UpdateInfoLabel();
    }
    else
    {
        delete zoomInfoLabel;
        delete posInfoLabel;
    }
}

void MainWidget::ToggleSave() 
{
    saveImg = !saveImg;
}

void MainWidget::ToggleNormalization()
{
    isNormalized = !isNormalized;
    BuildMandelbrot();
}

void MainWidget::Normalize()
{
    int minCol = 300, maxCol = 0, val;
    QRgb pixelData;
    QColor color;

    for ( int i = 0; i < imageWidth; i ++ )
        for ( int j = 0; j < imageHeight; j ++ )
        {
            pixelData = image -> pixel( i,j );
            color.setRgb( pixelData );
            val = color.red();
            minCol = std::min( minCol, val );
            maxCol = std::max( maxCol, val );
        }
    
    if( minCol != maxCol ) 
        for ( int i = 0; i < imageWidth; i ++ )
            for ( int j = 0; j < imageHeight; j ++ )
            {
                pixelData = image -> pixel( i,j );
                color.setRgb( pixelData );
                val = color.red();

                val = ( ( val - minCol ) * 255 ) / ( maxCol - minCol );

                image -> setPixel( i, j, qRgb( val, val, val ) );
            }
    else
        for ( int i = 0; i < imageWidth; i ++ )
            for ( int j = 0; j < imageHeight; j ++ )
                image -> setPixel( i, j, qRgb( 127, 127, 127 ) );
}

MainWidget::MainWidget( QWidget *parent ) :
    QWidget(parent)
{
    BuildUi();
    BuildMandelbrot();
    imgLabel -> installEventFilter(this);

    connect(pushReset,   SIGNAL(clicked()), this, SLOT(Reset()));
    connect(pushOptions, SIGNAL(clicked()), this, SLOT(OpenOptions()));
}
