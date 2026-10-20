param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$ReportPath
)
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $InputPath).Path
$reportFile = [IO.Path]::GetFullPath($ReportPath)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($reportFile)) | Out-Null
$existingPowerPoint = @(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$application = New-Object -ComObject PowerPoint.Application
$presentation = $null
$window = $null
$rows = @()
try {
    $originalSecurity = $application.AutomationSecurity
    $application.AutomationSecurity = 3
    $presentation = $application.Presentations.Open($sourcePath, -1, 0, 0)
    $presentation.SlideShowSettings.ShowType = 2
    $window = $presentation.SlideShowSettings.Run()
    # 窗口移至屏幕之外，仅检查本脚本创建的放映，不抢占用户全屏演示。
    $window.Left = -10000
    $window.Top = -10000
    $window.Width = 640
    $window.Height = 360
    foreach ($slide in $presentation.Slides) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.Type -ne 16) { continue }
            $window.View.GotoSlide($slide.SlideIndex)
            # 原生 PowerPoint 播放器：检查首段播放后时钟确实前进。
            # API: https://learn.microsoft.com/en-us/office/vba/api/powerpoint.slideshowview.player
            $player = $window.View.Player($shape.Id)
            $player.Play()
            $position = 0
            for ($attempt = 0; $attempt -lt 15; $attempt++) {
                Start-Sleep -Milliseconds 200
                $position = $player.CurrentPosition
                if ($position -gt 250) { break }
            }
            $row = [PSCustomObject]@{Slide=$slide.SlideIndex;Name=$shape.Name;LengthMs=$shape.MediaFormat.Length;PositionMs=$position;State=[int]$player.State;ClickAction=[int]$shape.ActionSettings(1).Action;Passed=($position -gt 250)}
            $rows += $row
            Write-Output ($row | ConvertTo-Json -Compress)
            $player.Stop()
        }
    }
    $rows | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 -LiteralPath $reportFile
    if ($rows.Count -ne 7 -or @($rows | Where-Object { -not $_.Passed }).Count -gt 0) {
        throw 'Native PowerPoint playback check did not pass all seven videos'
    }
} finally {
    # 某些版本在结束无编辑窗口的放映时会自动关闭文档/应用；清理应幂等。
    try { if ($null -ne $window) { $window.View.Exit() } } catch { Write-Verbose $_ }
    try { if ($null -ne $presentation) { $presentation.Close() } } catch { Write-Verbose $_ }
    try { if ($null -ne $originalSecurity) { $application.AutomationSecurity = $originalSecurity } } catch { Write-Verbose $_ }
    try { if (-not $existingPowerPoint) { $application.Quit() } } catch { Write-Verbose $_ }
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($application)
}
