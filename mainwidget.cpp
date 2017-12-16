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
#include <Qlabel>
#include <math.h>

void MainWidget::ZoomIn(const QPoint &p)  {
    QPointF pF = pixelCoordsToGraphCoords( p );
    double x = pF.x();
    double y = pF.y();
    xOrigin = x - ( x - xOrigin ) / zoomStep;
    yOrigin = y - ( y - yOrigin ) / zoomStep;
    zoomFactor *= zoomStep;
    BuildMandelbrot();
}

QPointF MainWidget::pixelCoordsToGraphCoords(const QPoint &p)
{
    double viewSize = (1 / zoomFactor) * 4;
    double xRatio = (double)p.x() / 600;
    double yRatio = (double)p.y() / 600;
    QPointF pF;
    pF.setX(xOrigin - viewSize / 2 + viewSize * xRatio);
    pF.setY(yOrigin - viewSize / 2 + viewSize * yRatio);

    return pF;
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
    //imgLabel -> setFocus(); //TODO: fix crash here

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
    image           = new QImage(600, 600, QImage::Format_RGB32);

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

QRgb MainWidget::Iterate(const double &xCoord, const double &yCoord)
{
    double x = xCoord, y = yCoord;
    for(int i = 1; i <= maxIterations; i ++)
    {
        if(x * x + y * y > 4)
        {
            int shade = 255 * (maxIterations - i + 1) / maxIterations;
            return qRgb(shade, shade, shade);
        }
        double xt = x * x - y * y + xCoord;
        double yt = 2 * x * y + yCoord;
        x = xt;
        y = yt;
    }
    return qRgb(0, 0, 0);
}

void MainWidget::BuildMandelbrot()
{
    for(int i = 0; i < 600; i ++ )
    {
        for ( int j = 0; j < 600; j ++ )
        {
            QPoint p( i, j );
            QPointF pF = pixelCoordsToGraphCoords( p );
            double x = pF.x(), y = pF.y();
            image -> setPixel( p, Iterate( x, y ) );
        }
    }

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

    for ( int i = 0; i < image -> height(); i ++ )
        for ( int j = 0; j < image -> width(); j ++ )
        {
            pixelData = image -> pixel( i,j );
            color.setRgb( pixelData );
            val = color.red();
            minCol = std::min( minCol, val );
            maxCol = std::max( maxCol, val );
        }
    
    if( minCol != maxCol ) 
        for ( int i = 0; i < image -> height(); i ++ )
            for ( int j = 0; j < image -> width(); j ++ )
            {
                pixelData = image -> pixel( i,j );
                color.setRgb( pixelData );
                val = color.red();

                val = ( ( val - minCol ) * 255 ) / ( maxCol - minCol );

                image -> setPixel( i, j, qRgb( val, val, val ) );
            }
    else
        for ( int i = 0; i < image -> height(); i ++ )
            for ( int j = 0; j < image -> width(); j ++ )
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
