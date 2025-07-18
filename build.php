<?php

include('vendor/autoload.php');

/**
 * Include here your custom libraries
 */

$slidey = new Gregwar\Slidey\Slidey;

/**
 * Customizing template
 */

$slidey->addCss('css/style.css');

// Sets the title prefix
$slidey->setTitle('Intro Python');

/**
 * Adding custom directories
 */

// This will copy the directory "css" to the target directory
$slidey->copy('css', 'css');
$slidey->copy('images', 'images');

// Adding zip files
$slidey
    ->copy('favicon.ico')
    ->mkdir('files')
    ->copy(__DIR__.'/files/*.zip', 'files/')
;

// Runs the build to the web directory
$target = 'web/';


$slidey->build($target);
