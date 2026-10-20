param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [switch]$ExportPdf
)
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $InputPath).Path
$targetPath = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($targetPath) | Out-Null
$existingPowerPoint = @(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$presentation = $null
$application = New-Object -ComObject PowerPoint.Application
try {
    # 只读、无窗口打开；禁用文档宏，不改变原演讲稿。
    $originalSecurity = $application.AutomationSecurity
    $application.AutomationSecurity = 3
    $presentation = $application.Presentations.Open($sourcePath, -1, 0, 0)
    $mediaCount = 0
    $layoutIssues = @()
    foreach ($slide in $presentation.Slides) {
        $slide.Export((Join-Path $targetPath ('slide_{0:D2}.png' -f $slide.SlideIndex)), 'PNG', 1600, 900)
        foreach ($shape in $slide.Shapes) {
            if ($shape.Type -eq 16) {
                $mediaCount++
                Write-Output ('MEDIA slide={0} name={1} type={2} duration_ms={3}' -f $slide.SlideIndex, $shape.Name, $shape.MediaType, $shape.MediaFormat.Length)
            }
            if ($shape.HasTextFrame -and $shape.TextFrame2.HasText) {
                $range = $shape.TextFrame2.TextRange
                if ($range.BoundHeight -gt $shape.Height + 2 -or $range.BoundWidth -gt $shape.Width + 2) {
                    $layoutIssues += [PSCustomObject]@{Slide=$slide.SlideIndex;Name=$shape.Name;Text=$range.Text;Height=$shape.Height;BoundHeight=$range.BoundHeight;Width=$shape.Width;BoundWidth=$range.BoundWidth}
                }
            }
        }
    }
    if ($ExportPdf) {
        $presentation.SaveAs((Join-Path $targetPath 'IBERO_V4_静态预览.pdf'), 32)
    }
    Write-Output ('RENDERED slides={0} media={1}' -f $presentation.Slides.Count, $mediaCount)
    ConvertTo-Json -InputObject @($layoutIssues) -Depth 5 | Set-Content -Encoding utf8 -LiteralPath (Join-Path $targetPath 'layout_issues.json')
    Write-Output ('TEXT_BOUND_ISSUES count={0}' -f $layoutIssues.Count)
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($null -ne $originalSecurity) { $application.AutomationSecurity = $originalSecurity }
    # 不退出用户原先打开的 PowerPoint；只关闭本脚本自己的文档。
    if (-not $existingPowerPoint) { $application.Quit() }
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($application)
}
