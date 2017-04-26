#include "optionswidget.h"
#include <QLabel>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QFrame>

void OptionsWidget::BuildUi() //crearea interfeței ferestrei de opțiuni
{
    //alocarea obiectelor pentru elementele de interfață
    instLabel = new QLabel( this );
    progbarCBox = new QCheckBox( this );
    infoCBox = new QCheckBox( this );
    normCbox = new QCheckBox( this );
    saveCbox = new QCheckBox( this );
    vbLay = new QVBoxLayout( this );
    separator = new QFrame( this );
    
    //configurarea elementelor
    instLabel -> setText( "Use left click to zoom in on an area.\nUse right click to zoom out.\nUse arow keys to move view.\nUse plus and minus keys to change the number of iterations.\nHold Shift for small increments or Control for large increments." );
    normCbox -> setText( "Normalize color" );
    progbarCBox -> setText( "Show build progress bar" );
    infoCBox -> setText( "Show view info" );
    saveCbox -> setText( "Automatically save image in application folder" );
    separator -> setFrameShape( QFrame::HLine );
    separator -> setFrameShadow( QFrame::Sunken );
    
    //creearea și setarea layout-ului
    vbLay -> addWidget( instLabel );
    vbLay -> addWidget( separator );
    vbLay -> addWidget( progbarCBox );
    vbLay -> addWidget( infoCBox );
    vbLay -> addWidget( normCbox );
    vbLay -> addWidget( saveCbox );
    vbLay -> setSizeConstraint( QLayout::SetFixedSize );
    setLayout( vbLay );
    setWindowFlags( windowFlags() ^ Qt::WindowMaximizeButtonHint ^ Qt::WindowStaysOnTopHint ); //flaguri pentru dezactivarea butonului de maximizare și păstrarea ferestrei deasupra celorlalte
}

void OptionsWidget::closeEvent( QCloseEvent* event ) //la închiderea ferestrei de opțiuni, cea principală este setată din nou activă
{
    parentWindow -> setEnabled( true );
}

//constructorul, primește în parametri opțiunile ferestrei principale pentru a inițializa bifele din casetele corespunzătoare
OptionsWidget::OptionsWidget( MainWidget* parent, bool progBar, bool info, bool norm, bool save ) :
    QWidget()
{
    parentWindow = parent; //memorarea adresei ferestrei principale (părinte) pentru utilizări în afara constructorului
    
    BuildUi();//creearea interfeței
    
    //setează bifele în funcție de opțiunile curente
    if ( progBar )
        progbarCBox-> setCheckState( Qt::Checked );
    if ( info )
        infoCBox -> setCheckState( Qt::Checked );;
    if ( norm )
        normCbox -> setCheckState( Qt::Checked );
    if ( save )
        saveCbox -> setCheckState( Qt::Checked );
    
    //conectează semnalele de la apăsarea casetelor de bifare la sloturile ferestrei principale pentru comutarea opțiunilor corespunzătoare
    connect( infoCBox, SIGNAL ( clicked() ), parent, SLOT( ToggleInfo() ) );
    connect( saveCbox, SIGNAL( clicked() ), parent, SLOT( ToggleSave() ) );
    connect( progbarCBox, SIGNAL( clicked() ), parent, SLOT( ToggleProgBar() ) );
    connect( normCbox, SIGNAL( clicked() ), parent, SLOT( ToggleNormalization() ) );
}
