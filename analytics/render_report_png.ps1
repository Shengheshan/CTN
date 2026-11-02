param([Parameter(Mandatory=$true)][string]$DataPath,[Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$data = Get-Content -LiteralPath $DataPath -Raw | ConvertFrom-Json
$bmp = New-Object System.Drawing.Bitmap(1600, 1200)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.Clear([System.Drawing.Color]::White)
$fontTitle = New-Object System.Drawing.Font('Arial', 12, [System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font('Arial', 8)
$penAxis = New-Object System.Drawing.Pen([System.Drawing.Color]::LightGray, 1)
$penLine = New-Object System.Drawing.Pen([System.Drawing.Color]::RoyalBlue, 2)
$brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::SteelBlue)
$dark = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Black)
$panelW = 515; $panelH = 370; $marginX = 20; $marginY = 20
for ($i=0; $i -lt $data.panels.Count; $i++) {
  $col = $i % 3; $row = [math]::Floor($i / 3)
  $x = $marginX + $col * 525; $y = $marginY + $row * 390
  $g.DrawRectangle($penAxis, $x, $y, $panelW, $panelH)
  $g.DrawString([string]$data.panels[$i].title, $fontTitle, $dark, $x+10, $y+8)
  $subRect = [System.Drawing.RectangleF]::new([single]($x+10),[single]($y+27),[single]($panelW-20),[single]32)
  $g.DrawString([string]$data.panels[$i].subtitle, $fontSmall, $dark, $subRect)
  $left=$x+38; $top=$y+60; $plotW=$panelW-55; $plotH=$panelH-108
  $g.DrawLine($penAxis, $left, $top+$plotH, $left+$plotW, $top+$plotH)
  $g.DrawLine($penAxis, $left, $top, $left, $top+$plotH)
  $points=@($data.panels[$i].points)
  if ($points.Count -eq 0) {
    $g.DrawString('Data insufficient', $fontSmall, $dark, $left+20, $top+25)
    continue
  }
  $values=@($points | ForEach-Object { [double]$_.value })
  $max=($values | Measure-Object -Maximum).Maximum
  if ($max -le 0) { $max=1 }
  $barW=[math]::Max(4, [math]::Floor($plotW / [math]::Max(1,$points.Count) * 0.58))
  for ($j=0; $j -lt $points.Count; $j++) {
    $slot=$plotW / [math]::Max(1,$points.Count)
    $h=[math]::Max(1, ([double]$points[$j].value / $max) * ($plotH-12))
    $bx=$left + $j*$slot + ($slot-$barW)/2
    $by=$top+$plotH-$h
    $g.FillRectangle($brush, [single]$bx, [single]$by, [single]$barW, [single]$h)
    if ($points.Count -le 12 -or ($j % [math]::Ceiling($points.Count/8) -eq 0)) {
      $label=[string]$points[$j].label
      if ($label.Length -gt 14) { $label=$label.Substring(0,14) }
      $g.DrawString($label, $fontSmall, $dark, [single]($bx-4), [single]($top+$plotH+4))
    }
  }
  $g.DrawString(('max {0:N2}' -f $max), $fontSmall, $dark, $left+4, $top+2)
  $linePoints=@($data.panels[$i].line_points)
  if ($linePoints.Count -gt 1) {
    for ($j=1; $j -lt $linePoints.Count; $j++) {
      $x1=$left + $plotW*($j-1)/($linePoints.Count-1)
      $x2=$left + $plotW*$j/($linePoints.Count-1)
      $v1=[double]$linePoints[$j-1].value; $v2=[double]$linePoints[$j].value
      $y1=$top+$plotH-($v1/$max)*($plotH-12)
      $y2=$top+$plotH-($v2/$max)*($plotH-12)
      $g.DrawLine($penLine,[single]$x1,[single]$y1,[single]$x2,[single]$y2)
    }
  }
}
$bmp.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose(); $fontTitle.Dispose(); $fontSmall.Dispose()
$penAxis.Dispose(); $penLine.Dispose(); $brush.Dispose(); $dark.Dispose()
