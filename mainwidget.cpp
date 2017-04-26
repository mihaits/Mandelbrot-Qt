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
#include <math.h>

void MainWidget::ZoomIn( const QPoint &p )  {
    QPointF pF = GraphCoord( p );
    double x = pF.x();
    double y = pF.y();
    xOrigin = x - ( x - xOrigin ) / zoomStep;
    yOrigin = y - ( y - yOrigin ) / zoomStep;
    zoomFactor *= zoomStep;
    BuildMandelbrot();
}

QPointF MainWidget::GraphCoord( const QPoint &p ) {
    double viewSize = ( 1 / zoomFactor ) * 4; //lățimea absolută a zonei ilustrate din fractal
    double xRatio = ( (double)( p.x() ) / 600 );
    double yRatio = ( (double)( p.y() ) / 600 );
    QPointF pF;
    pF.setX( xOrigin - viewSize / 2 + viewSize * xRatio );
    pF.setY(  ( yOrigin - viewSize / 2 + viewSize * yRatio ) );

    return pF;
}

bool MainWidget::eventFilter( QObject *obj, QEvent *event ) //filtrul de evenimente, efectuează instrucțiunile corespunzătoare pentru click-uri pe imagine și apăsarea anumitor taste
{
    imgLabel -> setFocus();
    //qDebug() << "orice eveniment";
    if ( obj == this -> imgLabel )  //qDebug() << " numai cele adresate imgLabel";
    {
        if ( event -> type() == QEvent::KeyPress )
        {
            //qDebug() << "daca e un eveniment de la tastatura";
            QKeyEvent* ev = static_cast<QKeyEvent*>( event );

            if ( ev -> key() == Qt::Key_Left )
            {
                double step = 1 / zoomFactor;
                if ( ev -> modifiers() & Qt::ShiftModifier )
                    step /= 10;
                xOrigin -= step;
                BuildMandelbrot();
                return true;
            }
            if ( ev -> key() == Qt::Key_Right )
            {
                double step = 1 / zoomFactor;
                if ( ev -> modifiers() & Qt::ShiftModifier )
                    step /= 10;
                xOrigin += step;
                BuildMandelbrot();
                return true;
            }
            if ( ev -> key() == Qt::Key_Up )
            {
                double step = 1 / zoomFactor;
                if ( ev -> modifiers() & Qt::ShiftModifier )
                    step /= 10;
                yOrigin -= step;
                BuildMandelbrot();
                return true;
            }
            if ( ev -> key() == Qt::Key_Down )
            {
                double step = 1 / zoomFactor;
                if ( ev -> modifiers() & Qt::ShiftModifier )
                    step /= 10;
                yOrigin += step;
                BuildMandelbrot();
                return true;
            }
            if ( ev -> key() == Qt::Key_Minus )
            {
                if( ev -> modifiers() & Qt::ControlModifier )
                {
                    if ( maxIterations > 100 )
                        maxIterations -= 100;
                }
                else
                {
                    if ( ev -> modifiers() & Qt::ShiftModifier )
                    {
                        if ( maxIterations > 1 )
                            maxIterations --;
                    }
                    else
                        if ( maxIterations > 10 )
                            maxIterations -= 10;
                }
                BuildMandelbrot();
                return true;
            }

            if ( ev -> key() == Qt::Key_Plus )
            {
                if ( ev -> modifiers() & Qt::ControlModifier )
                     maxIterations += 100;
                else
                {
                    if ( ev -> modifiers() & Qt::ShiftModifier )
                        maxIterations ++;
                    else
                        maxIterations += 10;
                }
                BuildMandelbrot();
                return true;
            }
        }
        //dacă e de la mouse
        if ( event -> type() == QEvent::MouseButtonRelease )
        {
            if ( isEnabled() ) //dacă fereastra e activă ( aparent doar partea asta merge și când e inactivă )
            {
                QMouseEvent* ev = static_cast<QMouseEvent*>( event );
                if ( ev -> button() == Qt::LeftButton )
                {
                    ZoomIn( ev -> pos() );
                    return true;
                }
                if ( ev -> button() == Qt::RightButton )
                {
                    ZoomOut();
                    return true;
                }
            }
        }
    }
    return false; //propagă semnalul mai departe
}

void MainWidget::UpdatePixmap() //QImage are funcții de editare directă pe pixeli și QPixmap e folosită cu QLabel pentru afișare deci ultimul pas la generarea unui fractal este convertirea din QImage în QPixmap și asocierea acestuia la label-ul folosit pentru afișare
{
    imgLabel -> setPixmap( QPixmap::fromImage( *image ) );
}

void MainWidget::UpdateInfoLabel()  //afișarea pe fereastră a nivelului de zoom și a numărului de iterații
{
    QString infoString;
    infoString.append( "Zoom: " );
    infoString.append( QString::number( zoomFactor ) );
    infoString.append( ". " );
    infoString.append( QString::number( maxIterations ) );
    infoString.append( " Iterations." );
    infoLabel -> setText( infoString );

    infoString.clear();
    infoString.append( "X = " );
    infoString.append( QString::number( xOrigin ) );
    infoString.append( ". Y = " );
    infoString.append( QString::number( - yOrigin ) );
    infoString.append( "." );
    infoLabel2 -> setText( infoString );
}

void MainWidget::BuildUi() //creearea interfeței
{
    //alocarea obiectelor pentru elemente de interfață
    infoLabel = new QLabel( this );
    infoLabel2 = new QLabel( this );
    pushReset = new QPushButton( this );
    pushOptions = new QPushButton( this );
    progBar = new QProgressBar( this );
    image = new QImage( 600, 600, QImage::Format_RGB32 );   //imaginea în care se creează fractalul
    imgLabel = new MyLabel;
    vbLay = new QVBoxLayout( this );
    hbLay = new QHBoxLayout( this );
    
    //configurarea obiectelor
    imgLabel -> setPixmap( QPixmap::fromImage( *image ) );  //soluția folosită în qt pentru a afișa imagini în fereastră
    pushReset -> setText( "Reset" );
    pushReset -> setFixedWidth( 100 );
    pushOptions -> setText( "Options and Instructions" );
    pushOptions -> setFixedWidth( 150 );
    progBar -> setRange( 0, 599 );
    progBar -> setTextVisible( false );
    progBar -> setFixedWidth( 600 );
    
    //construirea și setarea layout-urilor
    hbLay -> addStretch();
    hbLay -> addWidget( pushOptions );
    hbLay -> addWidget( pushReset );
    hbLay -> addStretch();

    vbLay -> addWidget( infoLabel, 0, Qt::AlignCenter );
    vbLay -> addWidget( infoLabel2, 0, Qt::AlignCenter );
    vbLay -> addLayout( hbLay );
    vbLay -> addWidget( imgLabel, 0, Qt::AlignCenter );
    vbLay -> addWidget( progBar, 0, Qt::AlignCenter );
    vbLay -> addStretch();
    vbLay -> setSizeConstraint( QLayout::SetFixedSize );
    setLayout( vbLay );

    setWindowFlags( windowFlags() ^ Qt::WindowMaximizeButtonHint ); //dezactivează butonul de maximise fereastra fiind de dimensiune fixă
}

void MainWidget::OpenOptions() //funcție slot care se apelează la apăsarea butonului de opțiuni și creează obiectul ferestrei respective
{
    OptionsWidget* optionsWindow;
    optionsWindow = new OptionsWidget( this, showProgbar, showInfo, isNormalized, saveImg ); //trasmite opțiunile curente în parametri
    optionsWindow -> show();
    
    //mută fereastra în centrul cele principale
    QPoint position = pos();
    position.setX( position.x() + width() / 2 - ( optionsWindow -> width() ) / 2 );
    position.setY( position.y() + height() / 2 - ( optionsWindow -> height() ) / 2 );

    optionsWindow -> move( position );

    setEnabled( false ); //dezactivează interacțiunea cu fereastra principală
}

QRgb MainWidget::Iterate( const double &a, const double &b )//funcția de iterare propriu-zisă, primește coordonate absolute în spațiul numerelor complexe și returnează o culoare din gradientul alb negru în funcție de numărul de iterări efectuate față de cel maxim
{
    double x = a, y = b;
    for ( int i = 1; i <= maxIterations; i ++ )
    {
        if ( x * x + y * y > 4 )
        {
            int shade = 255 * ( maxIterations - i + 1 ) / maxIterations;
            return qRgb ( shade, shade, shade );
        }
        double xt = x * x - y * y + a;
        double yt = 2 * x * y + b;
        x = xt;
        y = yt;
    }
    return qRgb( 0, 0, 0 );
}

void MainWidget::BuildMandelbrot() //funcția în care începe construirea fractalului
{
    for ( int i = 0; i < 600; i ++ )    //trece prin fiecare pixel din fereastră
    {
        if ( showProgbar && ( i + 1 ) % 10 == 0 ) //actualizează bara de progres dacă e cazul
            progBar -> setValue( i );
        for ( int j = 0; j < 600; j ++ )
        {
            QPoint p( i, j );
            QPointF pF = GraphCoord( p );   //obține coordonatele lui absolute
            double x = pF.x(), y = pF.y();
            image -> setPixel( p, Iterate( x, y ) );    //iterează pentru a obține culoarea lui
        }
    }

    if( isNormalized )      //aplicarea normalizării dacă e cazul
        Normalize();
    UpdatePixmap();         //actualizează imaginea
    if ( showInfo )
        UpdateInfoLabel();  //actualizează  textul informativ dacă e cazul
    if( saveImg )           //salvează imaginea în folderul aplicației dacă e cazul
    {
        QPixmap pixmap = QPixmap::fromImage( *image );
        pixmap.save( "image.jpg", 0, 90); 
    }
}

void MainWidget::ZoomOut()  //funcția de zoom out, scade zoomFactor în funcție de zoomStep și reconstruiește
{
    zoomFactor /= zoomStep;
    BuildMandelbrot();
}

void MainWidget::Reset() //resetează pozitia și numărul de iterații la valorile implicite și reconstruiește fractalul
{
    xOrigin = yOrigin = 0;
    zoomFactor = 1;
    maxIterations = 50;
    BuildMandelbrot();
}

void MainWidget::ToggleInfo() //funcție slot apelată la comutarea butonului pentru afișarea informațiilor
{
    showInfo = !showInfo;
    if( showInfo )
    {
        infoLabel = new QLabel( this );
        infoLabel2 = new QLabel( this );
        vbLay -> insertWidget( 0, infoLabel, 0, Qt::AlignCenter );
        vbLay -> insertWidget( 1, infoLabel2, 0, Qt::AlignCenter );
        UpdateInfoLabel();
    }
    else
    {
        delete infoLabel;
        delete infoLabel2;
    }
}

void MainWidget::ToggleSave() 
{
    saveImg = !saveImg;
}

void MainWidget::ToggleProgBar() //funcție slot apelată la comutarea butonului pentru afișarea barei de progres
{
    showProgbar = !showProgbar;
    if( showProgbar )
    {
        progBar = new QProgressBar( this );
        progBar -> setRange( 0, 599 );
        progBar -> setTextVisible( false );
        progBar -> setFixedWidth( 600 );
        vbLay -> addWidget( progBar, 0, Qt::AlignCenter );
    }
    else
        delete progBar;
}

void MainWidget::ToggleNormalization()  //funcție slot pentru comutarea normalizării, reconstruiește fractalul după aceasta
{
    isNormalized = !isNormalized;
    BuildMandelbrot();
}

void MainWidget::Normalize() //funcția de normalizare, extinde contrastul imaginii pentru a ocupa tot gradientul de la alb la negru
{
    int minCol = 300, maxCol = 0, val;
    QRgb pixeldata;
    QColor color;
    
    //găsește valorile minim și maxim
    for ( int i = 0; i < image -> height(); i ++ )
        for ( int j = 0; j < image -> width(); j ++ )
        {
            pixeldata = image -> pixel( i,j );
            color.setRgb( pixeldata );
            val = color.red();
            minCol = std::min( minCol, val );
            maxCol = std::max( maxCol, val );
        }
    
    if( minCol != maxCol ) 
        for ( int i = 0; i < image -> height(); i ++ )
            for ( int j = 0; j < image -> width(); j ++ )
            {
                pixeldata = image -> pixel( i,j );
                color.setRgb( pixeldata );
                val = color.red();

                val = ( ( val - minCol ) * 255 ) / ( maxCol - minCol );

                image -> setPixel( i, j, qRgb( val, val, val ) );
            }
    //dacă există o singură culoare face totul gri
    else
        for ( int i = 0; i < image -> height(); i ++ )
            for ( int j = 0; j < image -> width(); j ++ )
                image -> setPixel( i, j, qRgb( 127, 127, 127 ) );
}

//constructor
MainWidget::MainWidget( QWidget *parent ) :
    QWidget(parent)
{
    BuildUi(); //construiește interfața
    BuildMandelbrot(); //contruiește prima dată fractalul
    imgLabel -> installEventFilter( this ); //intalează filtrul de evenimente pe imgLabel
    
    //semnalele de la cele 2 butoane la sloturile corespunzătoare
    connect( pushReset, SIGNAL( clicked() ), this, SLOT( Reset() ) );
    connect( pushOptions, SIGNAL( clicked() ), this, SLOT( OpenOptions() ) );
}
