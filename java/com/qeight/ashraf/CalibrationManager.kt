package com.ashraf.qeight

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import org.json.JSONObject

/**
 * CalibrationManager — manages the ROI + Pool Table calibration overlay.
 *
 * Fix 1: No vignette/darkening — overlay is fully transparent except lines/circles.
 * Fix 2: Dual calibration mode (0 = ROI, 1 = Pool Table). Both rectangles drawn simultaneously.
 *         Pocket circles drawn on pool table rectangle only.
 * Fix 3: Pool table bounds saved under "pool_table_calibration_json".
 * Fix 4: Cancel broadcasts "com.ashraf.qeight.CALIBRATION_DISMISSED".
 *         No QeightJNI.setCalibrationMode() calls anywhere.
 *
 * Pocket N/S Shift parameter: corner pockets are always pinned exactly to the
 * pool-table rectangle's corners with no offset; only the two middle (rail)
 * pockets are shifted outward north/south, by [pocketNsShift].
 */
class CalibrationManager(
    private val context: Context,
    private val prefs: SharedPreferences
) {

    companion object {
        private const val TAG = "CalibrationManager"

        /**
         * Master switch for verbose logging.
         * Set to [true] to enable all Log.d / Log.i / Log.w / Log.e output.
         * Default is [false] (logging OFF) for production builds.
         */
        private const val LOGGING_ENABLED = false

        private fun logD(tag: String, msg: String) {
            if (LOGGING_ENABLED) Log.d(tag, msg)
        }
        private fun logI(tag: String, msg: String) {
            if (LOGGING_ENABLED) Log.i(tag, msg)
        }
        private fun logW(tag: String, msg: String, tr: Throwable? = null) {
            if (!LOGGING_ENABLED) return
            if (tr != null) Log.w(tag, msg, tr) else Log.w(tag, msg)
        }
        private fun logE(tag: String, msg: String, tr: Throwable? = null) {
            if (!LOGGING_ENABLED) return
            if (tr != null) Log.e(tag, msg, tr) else Log.e(tag, msg)
        }

        private const val KEY_ROI_CALIBRATION        = "roi_calibration_json"
        private const val KEY_POOL_TABLE_CALIBRATION = "pool_table_calibration_json"

        // Default ROI coordinates for 2400x1080 desktop reference — scaled to device on first use
        private const val DEFAULT_ROI_X1_REF = 442
        private const val DEFAULT_ROI_Y1_REF = 227
        private const val DEFAULT_ROI_X2_REF = 1958
        private const val DEFAULT_ROI_Y2_REF = 1012
        private const val DEFAULT_REF_W      = 2400
        private const val DEFAULT_REF_H      = 1080

        // Pocket defaults matching the reference C++ source
        private const val DEFAULT_POCKET_R      = 40
        // This constant controls ONLY the middle-pocket outward north/south shift.
        // Corner pockets are never offset — they are always drawn exactly at the
        // pool-table rectangle's corners.
        private const val DEFAULT_POCKET_NS_SHIFT = 30

        // Pool table default constants
        private const val DEFAULT_POOL_X1 = 470
        private const val DEFAULT_POOL_Y1 = 255
        private const val DEFAULT_POOL_X2 = 1930
        private const val DEFAULT_POOL_Y2 = 985

        // Hold-repeat interval for arrow buttons in milliseconds
        private const val HOLD_REPEAT_INTERVAL_MS = 80L

        // Target width:height ratio for the Pool Table rectangle, used by the
        // "Auto 2:1" adjustment button. Fixes the selected side and moves the
        // opposite side on the same axis so the rectangle becomes exactly this
        // ratio, using the current perpendicular-axis dimension as reference.
        private const val POOL_TABLE_TARGET_RATIO = 2.0

        private val HARDCODED_ROI_COORDS: IntArray? = null

        // ── Debug cushion overlay geometry (fixed real-world table proportions) ──
        private const val TABLE_LENGTH_CM = 254.0
        private const val TABLE_WIDTH_CM  = 127.0
        private const val POCKET_RADIUS_CM = 8.0

        private val CORNER_PROFILE = arrayOf(
            Pair(0.0, 10.6), Pair(-9.9, -0.6), Pair(-11.2, -5.7),
            Pair(-9.7, -9.7), Pair(-5.7, -11.2), Pair(-0.6, -9.9), Pair(10.6, 0.0)
        )
        private val SIDE_PROFILE = arrayOf(
            Pair(-7.9, 0.0), Pair(-6.2, -5.1), Pair(-5.8, -9.2), Pair(-3.8, -11.9),
            Pair(0.0, -13.2), Pair(3.8, -11.9), Pair(5.8, -9.2), Pair(6.2, -5.1), Pair(7.9, 0.0)
        )
        private val POCKET_CENTERS_CM = arrayOf(
            Pair(-130.8, -67.3), Pair(0.0, -71.0), Pair(130.8, -67.3),
            Pair(130.8, 67.3), Pair(0.0, 71.0), Pair(-130.8, 67.3)
        )

        // Builds the closed cushion polygon in table-centered cm space, mirroring
        // the corner/side pocket-mouth profiles into all 4 corners and both
        // mid-rail positions. Order: top-left corner, top-middle side (forward),
        // top-right corner (profile reversed), bottom-right corner (forward),
        // bottom-middle side (profile reversed), bottom-left corner (profile reversed).
        private fun buildCushionPolygonCm(): List<Pair<Double, Double>> {
            val pts = mutableListOf<Pair<Double, Double>>()
            val halfL = TABLE_LENGTH_CM / 2.0
            val halfW = TABLE_WIDTH_CM / 2.0

            for (c in CORNER_PROFILE) pts.add(Pair(c.first - halfL, c.second - halfW))
            for (s in SIDE_PROFILE)   pts.add(Pair(s.first, s.second - halfW))
            for (i in CORNER_PROFILE.indices.reversed()) {
                val c = CORNER_PROFILE[i]
                pts.add(Pair(-c.first + halfL, c.second - halfW))
            }
            for (c in CORNER_PROFILE) pts.add(Pair(-c.first + halfL, -c.second + halfW))
            for (i in SIDE_PROFILE.indices.reversed()) {
                val s = SIDE_PROFILE[i]
                pts.add(Pair(s.first, -s.second + halfW))
            }
            for (i in CORNER_PROFILE.indices.reversed()) {
                val c = CORNER_PROFILE[i]
                pts.add(Pair(c.first - halfL, -c.second + halfW))
            }
            return pts
        }
    }

    private val windowManager = context.getSystemService(Context.WINDOW_SERVICE)
            as WindowManager

    private val screenWidth: Int
    private val screenHeight: Int

    // ── ROI bounds ──────────────────────────────────────────────────────────
    private var roiX1 = DEFAULT_ROI_X1_REF
    private var roiY1 = DEFAULT_ROI_Y1_REF
    private var roiX2 = DEFAULT_ROI_X2_REF
    private var roiY2 = DEFAULT_ROI_Y2_REF

    // ── Pool table bounds ────────────────────────────────────────────────────
    private var poolX1 = DEFAULT_POOL_X1
    private var poolY1 = DEFAULT_POOL_Y1
    private var poolX2 = DEFAULT_POOL_X2
    private var poolY2 = DEFAULT_POOL_Y2

    // ── Pocket params ────────────────────────────────────────────────────────
    private var pocketR     = DEFAULT_POCKET_R
    // pocketNsShift: single parameter controlling the middle (rail) pocket
    //   outward north/south offset. Corner pockets are always pinned exactly
    //   to the pool-table rectangle's corners and are never affected by this.
    private var pocketNsShift = DEFAULT_POCKET_NS_SHIFT

    // ── Debug overlay state ──────────────────────────────────────────────────
    private var showCushionDebug = false

    // ── UI state ─────────────────────────────────────────────────────────────
    /**
     * calibMode:
     *   0 = editing ROI bounds
     *   1 = editing Pool Table bounds
     */
    private var calibMode    = 0
    private var selectedSide = 0   // 0=LEFT, 1=RIGHT, 2=TOP, 3=BOTTOM

    private var calibrationView: CalibrationView? = null
    private var calibrationViewParams: WindowManager.LayoutParams? = null

    private var controlsBar: LinearLayout? = null
    private var controlsBarParams: WindowManager.LayoutParams? = null

    private val repeatHandler = Handler(Looper.getMainLooper())

    private val sideButtons = arrayOfNulls<Button>(4)
    private val modeButtons = arrayOfNulls<Button>(2)   // [0]=ROI, [1]=Pool Table

    private var btnArrow1: Button? = null
    private var btnArrow2: Button? = null

    // ── Pool-table-only auto-adjust / copy row ──────────────────────────────
    private var btnAutoRatio: Button? = null
    private var btnCopyCoords: Button? = null
    private var poolToolsRow: LinearLayout? = null

    // ── Title text view (updated when mode changes) ──────────────────────────
    private var titleText: TextView? = null

    // Always-visible readout of live pool table coords — lives in the same
    // window as the buttons, so unlike a Toast it cannot be drawn behind
    // other overlay layers. Updated any time pool table bounds change.
    private var poolCoordsReadout: TextView? = null

    init {
        val metrics = android.util.DisplayMetrics()
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getRealMetrics(metrics)
        screenWidth  = maxOf(metrics.widthPixels, metrics.heightPixels)
        screenHeight = minOf(metrics.widthPixels, metrics.heightPixels)

        loadFromPrefs()
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Persistence
    // ────────────────────────────────────────────────────────────────────────

    private fun loadFromPrefs() {
        try {
            // ── ROI ──────────────────────────────────────────────────────────
            HARDCODED_ROI_COORDS?.let { coords ->
                if (coords.size == 4) {
                    roiX1 = coords[0]; roiY1 = coords[1]
                    roiX2 = coords[2]; roiY2 = coords[3]
                    pocketR       = DEFAULT_POCKET_R
                    pocketNsShift = DEFAULT_POCKET_NS_SHIFT
                    logI(TAG, "loadFromPrefs: HARDCODED ROI=($roiX1,$roiY1,$roiX2,$roiY2)")
                    // Still try to load pool table from prefs when using hardcoded ROI
                    loadPoolTableFromPrefs()
                    return
                }
            }

            val roiJson = prefs.getString(KEY_ROI_CALIBRATION, null)
            if (roiJson != null) {
                val obj = JSONObject(roiJson)
                roiX1       = obj.getInt("roi_x1")
                roiY1       = obj.getInt("roi_y1")
                roiX2       = obj.getInt("roi_x2")
                roiY2       = obj.getInt("roi_y2")
                pocketR     = obj.optInt("pocket_r",     DEFAULT_POCKET_R)
                // Try the current key first, then fall back to previously used
                // key names ("pocket_inset", then "pocket_ns") for backwards
                // compatibility with previously saved calibration data.
                pocketNsShift = obj.optInt("pocket_ns_shift",
                    obj.optInt("pocket_inset",
                        obj.optInt("pocket_ns", DEFAULT_POCKET_NS_SHIFT)))
            } else {
                // Scale defaults from reference resolution
                roiX1 = (DEFAULT_ROI_X1_REF * screenWidth  / DEFAULT_REF_W.toFloat()).toInt()
                roiY1 = (DEFAULT_ROI_Y1_REF * screenHeight / DEFAULT_REF_H.toFloat()).toInt()
                roiX2 = (DEFAULT_ROI_X2_REF * screenWidth  / DEFAULT_REF_W.toFloat()).toInt()
                roiY2 = (DEFAULT_ROI_Y2_REF * screenHeight / DEFAULT_REF_H.toFloat()).toInt()
                pocketR       = DEFAULT_POCKET_R
                pocketNsShift = DEFAULT_POCKET_NS_SHIFT
            }

            loadPoolTableFromPrefs()

        } catch (e: Exception) {
            logE(TAG, "loadFromPrefs: Exception — using defaults", e)
        }
    }

    /**
     * Loads pool table bounds from prefs. Falls back to scaled pool table defaults when absent.
     */
    private fun loadPoolTableFromPrefs() {
        try {
            val poolJson = prefs.getString(KEY_POOL_TABLE_CALIBRATION, null)
            if (poolJson != null) {
                val obj = JSONObject(poolJson)
                poolX1 = obj.getInt("pool_x1")
                poolY1 = obj.getInt("pool_y1")
                poolX2 = obj.getInt("pool_x2")
                poolY2 = obj.getInt("pool_y2")
            } else {
                // Scale 2400×1080 reference defaults to actual device landscape resolution
                poolX1 = (DEFAULT_POOL_X1 * screenWidth  / DEFAULT_REF_W.toFloat()).toInt()
                poolY1 = (DEFAULT_POOL_Y1 * screenHeight / DEFAULT_REF_H.toFloat()).toInt()
                poolX2 = (DEFAULT_POOL_X2 * screenWidth  / DEFAULT_REF_W.toFloat()).toInt()
                poolY2 = (DEFAULT_POOL_Y2 * screenHeight / DEFAULT_REF_H.toFloat()).toInt()
            }
        } catch (e: Exception) {
            logE(TAG, "loadPoolTableFromPrefs: Exception — using scaled defaults", e)
            poolX1 = (DEFAULT_POOL_X1 * screenWidth  / DEFAULT_REF_W.toFloat()).toInt()
            poolY1 = (DEFAULT_POOL_Y1 * screenHeight / DEFAULT_REF_H.toFloat()).toInt()
            poolX2 = (DEFAULT_POOL_X2 * screenWidth  / DEFAULT_REF_W.toFloat()).toInt()
            poolY2 = (DEFAULT_POOL_Y2 * screenHeight / DEFAULT_REF_H.toFloat()).toInt()
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Public API
    // ────────────────────────────────────────────────────────────────────────

    fun startCalibration() {
        try {
            addCalibrationView()
            addControlsBar()
        } catch (e: Exception) {
            logE(TAG, "startCalibration: Exception", e)
            Toast.makeText(context, "Failed to start calibration: ${e.message}",
                Toast.LENGTH_SHORT).show()
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Overlay views
    // ────────────────────────────────────────────────────────────────────────

    private fun addCalibrationView() {
        calibrationView = CalibrationView(context)

        val params = WindowManager.LayoutParams(
            screenWidth,
            screenHeight,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 0
            y = 0
        }

        calibrationViewParams = params
        windowManager.addView(calibrationView, params)
    }

    private fun addControlsBar() {
        val density = context.resources.displayMetrics.density
        val padding = (10 * density).toInt()

        val panelW = minOf(
            (screenWidth * 0.28f).toInt(),
            (280 * density).toInt()
        )

        // ── Outer container (direct child of WindowManager) ──────────────────
        controlsBar = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#CC000000"))
            setPadding(padding, padding, padding, padding)
        }

        // ── Title row — fixed header / drag handle ───────────────────────────
        val titleRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.parseColor("#FF6D00"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        titleText = TextView(context).apply {
            text = modeTitle()
            setTextColor(Color.WHITE)
            textSize = 10f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(
                (6 * density).toInt(), (6 * density).toInt(),
                (6 * density).toInt(), (6 * density).toInt()
            )
        }
        titleRow.addView(titleText)

        controlsBar?.addView(titleRow)

        // ── Scrollable content ───────────────────────────────────────────────
        val scrollContent = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 0, 0, (8 * density).toInt())
        }

        val scrollView = ScrollView(context).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
            )
            addView(scrollContent)
        }

        controlsBar?.addView(scrollView)

        // ── Mode toggle row (ROI | Pool Table) ──────────────────────────────
        val modeRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (8 * density).toInt() }
        }

        val modeLabels = arrayOf("ROI", "Pool Table")
        modeLabels.forEachIndexed { index, label ->
            val btn = Button(context).apply {
                text = label
                textSize = 9f
                setTextColor(Color.WHITE)
                layoutParams = LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
                ).apply {
                    if (index == 0) marginEnd = (4 * density).toInt()
                }
                setOnClickListener {
                    calibMode = index
                    updateModeButtonColors()
                    updateArrowButtons()
                    updatePoolToolsVisibility()
                    titleText?.text = modeTitle()
                    calibrationView?.invalidate()
                }
            }
            modeButtons[index] = btn
            modeRow.addView(btn)
        }
        scrollContent.addView(modeRow)
        updateModeButtonColors()

        scrollContent.addView(makeDivider(density))

        // ── Side selector row ────────────────────────────────────────────────
        val sideRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (4 * density).toInt() }
        }

        val sideLabels = arrayOf("LEFT", "RIGHT", "TOP", "BOTTOM")
        sideLabels.forEachIndexed { index, label ->
            val btn = Button(context).apply {
                text = label
                textSize = 9f
                setTextColor(Color.WHITE)
                layoutParams = LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
                ).apply {
                    if (index < 3) marginEnd = (4 * density).toInt()
                }
                setOnClickListener {
                    selectedSide = index
                    updateSideButtonColors()
                    updateArrowButtons()
                }
            }
            sideButtons[index] = btn
            sideRow.addView(btn)
        }
        scrollContent.addView(sideRow)
        updateSideButtonColors()

        // ── Arrow buttons ────────────────────────────────────────────────────
        val arrowRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin    = (8 * density).toInt()
                bottomMargin = (4 * density).toInt()
            }
        }

        btnArrow1 = Button(context).apply {
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#FF6D00"))
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            ).apply { marginEnd = (8 * density).toInt() }
        }

        btnArrow2 = Button(context).apply {
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#FF6D00"))
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            )
        }

        arrowRow.addView(btnArrow1)
        arrowRow.addView(btnArrow2)
        scrollContent.addView(arrowRow)
        updateArrowButtons()

        scrollContent.addView(makeDivider(density))

        // ── Pocket param spinners ────────────────────────────────────────────
        addCalibSpinnerRow(
            container = scrollContent,
            label     = "Pocket R",
            initial   = pocketR,
            min       = 5,
            max       = 200,
            onGet     = { pocketR },
            onSet     = { v ->
                pocketR = v
                calibrationView?.invalidate()
                try { QeightJNI.setPocketParams(pocketR, pocketNsShift) } catch (_: Exception) {}
            }
        )

        addCalibSpinnerRow(
            container = scrollContent,
            label     = "Pocket N/S Shift",
            initial   = pocketNsShift,
            min       = 0,
            max       = 300,
            onGet     = { pocketNsShift },
            onSet     = { v ->
                pocketNsShift = v
                calibrationView?.invalidate()
                try { QeightJNI.setPocketParams(pocketR, pocketNsShift) } catch (_: Exception) {}
            }
        )

        scrollContent.addView(makeDivider(density))

        // ── Debug: Cushion Outline checkbox ──────────────────────────────────
        val debugRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, (4 * density).toInt(), 0, (4 * density).toInt())
        }
        debugRow.addView(TextView(context).apply {
            text = "Debug: Cushion Outline"
            setTextColor(Color.WHITE)
            textSize = 10.5f
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        })
        debugRow.addView(android.widget.CheckBox(context).apply {
            isChecked = showCushionDebug
            setOnCheckedChangeListener { _, checked ->
                showCushionDebug = checked
                calibrationView?.invalidate()
            }
        })
        scrollContent.addView(debugRow)
        scrollContent.addView(makeDivider(density))

        // ── Pool-table-only tools: Auto 2:1 ratio + Copy Coords ──────────────
        // Placed directly above SAVE/CANCEL (a known-reachable, always-tappable
        // area) so it can't be scrolled out of reach or hidden behind anything.
        poolToolsRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            elevation = 8f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (6 * density).toInt() }
        }

        btnAutoRatio = Button(context).apply {
            text = "AUTO 2:1"
            textSize = 11f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#0077CC"))
            isClickable = true
            isFocusable = true
            layoutParams = LinearLayout.LayoutParams(
                0, (44 * density).toInt(), 1f
            ).apply { marginEnd = (4 * density).toInt() }
            setOnClickListener {
                logI(TAG, "btnAutoRatio: clicked, calibMode=$calibMode selectedSide=$selectedSide")
                applyAutoRatioToPoolTable()
            }
        }

        btnCopyCoords = Button(context).apply {
            text = "COPY COORDS"
            textSize = 11f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#444444"))
            isClickable = true
            isFocusable = true
            layoutParams = LinearLayout.LayoutParams(
                0, (44 * density).toInt(), 1f
            )
            setOnClickListener { copyPoolTableCoordsToClipboard() }
        }

        poolToolsRow?.addView(btnAutoRatio)
        poolToolsRow?.addView(btnCopyCoords)
        scrollContent.addView(poolToolsRow)

        poolCoordsReadout = TextView(context).apply {
            text = poolCoordsText()
            setTextColor(Color.parseColor("#FFE082"))
            textSize = 10f
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (4 * density).toInt() }
        }
        scrollContent.addView(poolCoordsReadout)
        updatePoolToolsVisibility()

        scrollContent.addView(makeDivider(density))

        // ── Action row (SAVE / CANCEL) ───────────────────────────────────────
        val actionRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (6 * density).toInt() }
        }

        val btnSave = Button(context).apply {
            text = "SAVE"
            setTextColor(Color.BLACK)
            setBackgroundColor(Color.parseColor("#FF8C00"))
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            )
            setOnClickListener { saveAndExit() }
        }

        val actionSpacer = View(context).apply {
            layoutParams = LinearLayout.LayoutParams(
                (8 * density).toInt(), LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val btnCancel = Button(context).apply {
            text = "CANCEL"
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#333333"))
            layoutParams = LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
            )
            setOnClickListener { onCancelButtonPressed() }
        }

        actionRow.addView(btnSave)
        actionRow.addView(actionSpacer)
        actionRow.addView(btnCancel)
        scrollContent.addView(actionRow)

        // ── WindowManager params ─────────────────────────────────────────────
        val controlsParams = WindowManager.LayoutParams(
            panelW,
            (screenHeight * 0.82f).toInt(),
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = (8 * density).toInt()
            y = (screenHeight * 0.09f).toInt()
            screenOrientation = ActivityInfo.SCREEN_ORIENTATION_USER_LANDSCAPE
        }

        controlsBarParams = controlsParams
        windowManager.addView(controlsBar, controlsParams)

        // ── Drag-to-move on titleRow ─────────────────────────────────────────
        var dragStartRawX = 0f
        var dragStartRawY = 0f
        var dragParamX    = 0
        var dragParamY    = 0

        titleRow.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    dragStartRawX = event.rawX
                    dragStartRawY = event.rawY
                    dragParamX = controlsBarParams?.x ?: 0
                    dragParamY = controlsBarParams?.y ?: 0
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (event.rawX - dragStartRawX).toInt()
                    val dy = (event.rawY - dragStartRawY).toInt()
                    controlsBarParams?.x = (dragParamX + dx).coerceIn(0, screenWidth - panelW)
                    controlsBarParams?.y = (dragParamY + dy).coerceIn(0, screenHeight - 200)
                    try {
                        if (controlsBar?.parent != null)
                            windowManager.updateViewLayout(controlsBar, controlsBarParams)
                    } catch (_: Exception) {}
                    true
                }
                else -> false
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  UI helpers
    // ────────────────────────────────────────────────────────────────────────

    private fun modeTitle() = when (calibMode) {
        1    -> "POOL TABLE CALIBRATION"
        else -> "ROI CALIBRATION"
    }

    private fun updateModeButtonColors() {
        val active   = Color.parseColor("#0077CC")
        val roiColor = Color.parseColor("#FF6D00")
        val inactive = Color.parseColor("#444444")

        modeButtons[0]?.setBackgroundColor(if (calibMode == 0) roiColor else inactive)
        modeButtons[1]?.setBackgroundColor(if (calibMode == 1) active   else inactive)
    }

    private fun updateSideButtonColors() {
        val selected   = Color.parseColor("#FF6D00")
        val unselected = Color.parseColor("#444444")
        sideButtons.forEachIndexed { index, btn ->
            btn?.setBackgroundColor(if (index == selectedSide) selected else unselected)
        }
    }

    /**
     * The Auto 2:1 / Copy Coords row only applies to the Pool Table rectangle,
     * so it's hidden entirely while editing the ROI.
     */
    private fun updatePoolToolsVisibility() {
        val vis = if (calibMode == 1) View.VISIBLE else View.GONE
        poolToolsRow?.visibility = vis
        poolCoordsReadout?.visibility = vis
        refreshPoolCoordsReadout()
    }

    private fun poolCoordsText(): String {
        val ratio = if (poolY2 != poolY1)
            (poolX2 - poolX1).toDouble() / (poolY2 - poolY1).toDouble()
        else 0.0
        return "Pool: ($poolX1,$poolY1,$poolX2,$poolY2)  ratio=${String.format("%.4f", ratio)}:1"
    }

    /** Refreshes the always-visible readout — call after any pool table mutation. */
    private fun refreshPoolCoordsReadout() {
        poolCoordsReadout?.text = poolCoordsText()
    }

    /**
     * Fixes the currently-selected side of the POOL TABLE rectangle and moves
     * the opposite side on the same axis so that width:height becomes exactly
     * [POOL_TABLE_TARGET_RATIO] : 1, using the current perpendicular-axis
     * dimension as the reference (it is left untouched).
     *
     *   LEFT/RIGHT selected   → height is reference, width solved as 2×height
     *   TOP/BOTTOM selected    → width is reference, height solved as width/2
     *
     * The computed opposite edge is clamped to screen bounds / the fixed edge,
     * same as the arrow-button limits. If clamping kicks in, the result will
     * be as close to 2:1 as the screen allows and the user is toasted about it.
     */
    private fun applyAutoRatioToPoolTable() {
        if (calibMode != 1) return

        val exact: Boolean
        when (selectedSide) {
            0 -> {   // LEFT fixed → solve RIGHT from current height
                val height = (poolY2 - poolY1).toDouble()
                val desiredX2 = poolX1 + (height * POOL_TABLE_TARGET_RATIO)
                val clamped = desiredX2.toInt().coerceIn(poolX1 + 1, screenWidth)
                exact = clamped == desiredX2.toInt()
                poolX2 = clamped
            }
            1 -> {   // RIGHT fixed → solve LEFT from current height
                val height = (poolY2 - poolY1).toDouble()
                val desiredX1 = poolX2 - (height * POOL_TABLE_TARGET_RATIO)
                val clamped = desiredX1.toInt().coerceIn(0, poolX2 - 1)
                exact = clamped == desiredX1.toInt()
                poolX1 = clamped
            }
            2 -> {   // TOP fixed → solve BOTTOM from current width
                val width = (poolX2 - poolX1).toDouble()
                val desiredY2 = poolY1 + (width / POOL_TABLE_TARGET_RATIO)
                val clamped = desiredY2.toInt().coerceIn(poolY1 + 1, screenHeight)
                exact = clamped == desiredY2.toInt()
                poolY2 = clamped
            }
            else -> { // BOTTOM fixed → solve TOP from current width
                val width = (poolX2 - poolX1).toDouble()
                val desiredY1 = poolY2 - (width / POOL_TABLE_TARGET_RATIO)
                val clamped = desiredY1.toInt().coerceIn(0, poolY2 - 1)
                exact = clamped == desiredY1.toInt()
                poolY1 = clamped
            }
        }

        calibrationView?.invalidate()
        refreshPoolCoordsReadout()

        if (!exact) {
            Toast.makeText(
                context,
                "Reached screen edge — ratio not exactly 2:1",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    /**
     * Copies the current Pool Table rectangle coordinates to the system
     * clipboard as plain text, e.g. "poolX1=469, poolY1=258, poolX2=1929, poolY2=983".
     */
    private fun copyPoolTableCoordsToClipboard() {
        val text = "poolX1=$poolX1, poolY1=$poolY1, poolX2=$poolX2, poolY2=$poolY2"
        try {
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            clipboard.setPrimaryClip(ClipData.newPlainText("Pool Table Coordinates", text))
            Toast.makeText(context, "Copied: $text", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            logE(TAG, "copyPoolTableCoordsToClipboard: Exception", e)
            Toast.makeText(context, "Failed to copy coordinates", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateArrowButtons() {
        btnArrow1?.setOnTouchListener(null)
        btnArrow2?.setOnTouchListener(null)

        when (selectedSide) {
            0 -> {   // LEFT edge
                btnArrow1?.text = "◄  Move Left"
                btnArrow2?.text = "Move Right  ►"
                if (calibMode == 0) {
                    attachArrowHoldRepeat(btnArrow1!!) {
                        roiX1 = (roiX1 - 1).coerceIn(0, roiX2 - 1)
                        calibrationView?.invalidate()
                    }
                    attachArrowHoldRepeat(btnArrow2!!) {
                        roiX1 = (roiX1 + 1).coerceIn(0, roiX2 - 1)
                        calibrationView?.invalidate()
                    }
                } else {
                    attachArrowHoldRepeat(btnArrow1!!) {
                        poolX1 = (poolX1 - 1).coerceIn(0, poolX2 - 1)
                        calibrationView?.invalidate()
                        refreshPoolCoordsReadout()
                    }
                    attachArrowHoldRepeat(btnArrow2!!) {
                        poolX1 = (poolX1 + 1).coerceIn(0, poolX2 - 1)
                        calibrationView?.invalidate()
                        refreshPoolCoordsReadout()
                    }
                }
            }
            1 -> {   // RIGHT edge
                btnArrow1?.text = "◄  Move Left"
                btnArrow2?.text = "Move Right  ►"
                if (calibMode == 0) {
                    attachArrowHoldRepeat(btnArrow1!!) {
                        roiX2 = (roiX2 - 1).coerceIn(roiX1 + 1, screenWidth)
                        calibrationView?.invalidate()
                    }
                    attachArrowHoldRepeat(btnArrow2!!) {
                        roiX2 = (roiX2 + 1).coerceIn(roiX1 + 1, screenWidth)
                        calibrationView?.invalidate()
                    }
                } else {
                    attachArrowHoldRepeat(btnArrow1!!) {
                        poolX2 = (poolX2 - 1).coerceIn(poolX1 + 1, screenWidth)
                        calibrationView?.invalidate()
                        refreshPoolCoordsReadout()
                    }
                    attachArrowHoldRepeat(btnArrow2!!) {
                        poolX2 = (poolX2 + 1).coerceIn(poolX1 + 1, screenWidth)
                        calibrationView?.invalidate()
                        refreshPoolCoordsReadout()
                    }
                }
            }
            2 -> {   // TOP edge
                btnArrow1?.text = "▲  Move Up"
                btnArrow2?.text = "Move Down  ▼"
                if (calibMode == 0) {
                    attachArrowHoldRepeat(btnArrow1!!) {
                        roiY1 = (roiY1 - 1).coerceIn(0, roiY2 - 1)
                        calibrationView?.invalidate()
                    }
                    attachArrowHoldRepeat(btnArrow2!!) {
                        roiY1 = (roiY1 + 1).coerceIn(0, roiY2 - 1)
                        calibrationView?.invalidate()
                    }
                } else {
                    attachArrowHoldRepeat(btnArrow1!!) {
                        poolY1 = (poolY1 - 1).coerceIn(0, poolY2 - 1)
                        calibrationView?.invalidate()
                        refreshPoolCoordsReadout()
                    }
                    attachArrowHoldRepeat(btnArrow2!!) {
                        poolY1 = (poolY1 + 1).coerceIn(0, poolY2 - 1)
                        calibrationView?.invalidate()
                        refreshPoolCoordsReadout()
                    }
                }
            }
            3 -> {   // BOTTOM edge
                btnArrow1?.text = "▲  Move Up"
                btnArrow2?.text = "Move Down  ▼"
                if (calibMode == 0) {
                    attachArrowHoldRepeat(btnArrow1!!) {
                        roiY2 = (roiY2 - 1).coerceIn(roiY1 + 1, screenHeight)
                        calibrationView?.invalidate()
                    }
                    attachArrowHoldRepeat(btnArrow2!!) {
                        roiY2 = (roiY2 + 1).coerceIn(roiY1 + 1, screenHeight)
                        calibrationView?.invalidate()
                    }
                } else {
                    attachArrowHoldRepeat(btnArrow1!!) {
                        poolY2 = (poolY2 - 1).coerceIn(poolY1 + 1, screenHeight)
                        calibrationView?.invalidate()
                        refreshPoolCoordsReadout()
                    }
                    attachArrowHoldRepeat(btnArrow2!!) {
                        poolY2 = (poolY2 + 1).coerceIn(poolY1 + 1, screenHeight)
                        calibrationView?.invalidate()
                        refreshPoolCoordsReadout()
                    }
                }
            }
        }
    }

    private fun makeDivider(density: Float): View = View(context).apply {
        setBackgroundColor(Color.parseColor("#555555"))
        layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            (1 * density).toInt()
        ).apply {
            topMargin    = (6 * density).toInt()
            bottomMargin = (6 * density).toInt()
        }
    }

    private fun addCalibSpinnerRow(
        container: LinearLayout,
        label: String,
        initial: Int,
        min: Int,
        max: Int,
        onGet: () -> Int,
        onSet: (Int) -> Unit
    ): TextView {
        val density = context.resources.displayMetrics.density

        val row = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, (4 * density).toInt(), 0, (4 * density).toInt())
        }

        val labelView = TextView(context).apply {
            text = label
            setTextColor(Color.WHITE)
            textSize = 11f
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val btnMinus = Button(context).apply {
            text = "−"
            textSize = 14f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#444444"))
            layoutParams = LinearLayout.LayoutParams(
                (36 * density).toInt(), (36 * density).toInt()
            )
        }

        val valueView = TextView(context).apply {
            text = initial.toString()
            setTextColor(Color.WHITE)
            textSize = 13f
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                (48 * density).toInt(), LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val btnPlus = Button(context).apply {
            text = "+"
            textSize = 14f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#444444"))
            layoutParams = LinearLayout.LayoutParams(
                (36 * density).toInt(), (36 * density).toInt()
            )
        }

        btnMinus.setOnClickListener {
            val cur = onGet()
            if (cur > min) { onSet(cur - 1); valueView.text = onGet().toString() }
        }
        btnPlus.setOnClickListener {
            val cur = onGet()
            if (cur < max) { onSet(cur + 1); valueView.text = onGet().toString() }
        }

        row.addView(labelView)
        row.addView(btnMinus)
        row.addView(valueView)
        row.addView(btnPlus)
        container.addView(row)

        return valueView
    }

    private fun attachArrowHoldRepeat(button: Button, action: () -> Unit) {
        var isHeld = false
        val repeatRunnable = object : Runnable {
            override fun run() {
                if (isHeld) {
                    action()
                    repeatHandler.postDelayed(this, HOLD_REPEAT_INTERVAL_MS)
                }
            }
        }
        button.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    isHeld = true
                    action()
                    repeatHandler.postDelayed(repeatRunnable, HOLD_REPEAT_INTERVAL_MS)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    isHeld = false
                    repeatHandler.removeCallbacks(repeatRunnable)
                    true
                }
                else -> false
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Save / Dismiss
    // ────────────────────────────────────────────────────────────────────────

    private fun saveAndExit() {
        try {
            // ── Persist ROI ──────────────────────────────────────────────────
            // Save under the key name "pocket_ns_shift". Old "pocket_inset" and
            // "pocket_ns" keys are no longer written; reading still falls back
            // to them for compatibility.
            val roiJson = JSONObject().apply {
                put("roi_x1",          roiX1)
                put("roi_y1",          roiY1)
                put("roi_x2",          roiX2)
                put("roi_y2",          roiY2)
                put("pocket_r",        pocketR)
                put("pocket_ns_shift", pocketNsShift)
            }.toString()

            // ── Persist Pool Table ───────────────────────────────────────────
            val poolJson = JSONObject().apply {
                put("pool_x1", poolX1)
                put("pool_y1", poolY1)
                put("pool_x2", poolX2)
                put("pool_y2", poolY2)
            }.toString()

            prefs.edit()
                .putString(KEY_ROI_CALIBRATION,        roiJson)
                .putString(KEY_POOL_TABLE_CALIBRATION,  poolJson)
                .apply()

            // ── Push live to native layer ────────────────────────────────────
            try { QeightJNI.setRoi(roiX1, roiY1, roiX2, roiY2)                   } catch (_: Exception) {}
            try { QeightJNI.setPocketParams(pocketR, pocketNsShift)               } catch (_: Exception) {}
            try { QeightJNI.setPoolTableBounds(poolX1, poolY1, poolX2, poolY2)    } catch (_: Exception) {}

            val msg = "ROI: ($roiX1,$roiY1,$roiX2,$roiY2)  Pool: ($poolX1,$poolY1,$poolX2,$poolY2)"
            Toast.makeText(context, msg, Toast.LENGTH_LONG).show()

            LocalBroadcastManager.getInstance(context).sendBroadcast(
                Intent("com.ashraf.qeight.CALIBRATION_SAVED")
            )

            dismiss()

        } catch (e: Exception) {
            Toast.makeText(context, "Failed to save: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Called by the Cancel button.
     * Sends CALIBRATION_DISMISSED broadcast then tears down the overlay.
     */
    private fun onCancelButtonPressed() {
        LocalBroadcastManager.getInstance(context).sendBroadcast(
            Intent("com.ashraf.qeight.CALIBRATION_DISMISSED")
        )
        dismiss()
    }

    /**
     * Public teardown method. Called by OverlayService.onDestroy() to clean up
     * overlay views when the service stops. Also called internally by
     * saveAndExit() and onCancelButtonPressed().
     */
    fun dismiss() {
        try {
            repeatHandler.removeCallbacksAndMessages(null)
            if (calibrationView?.parent != null) windowManager.removeView(calibrationView)
            if (controlsBar?.parent     != null) windowManager.removeView(controlsBar)
            calibrationView = null
            controlsBar     = null
        } catch (e: Exception) {
            logE(TAG, "dismiss: Exception", e)
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  CalibrationView — the transparent overlay canvas
    // ────────────────────────────────────────────────────────────────────────

    inner class CalibrationView(context: Context) : View(context) {

        // ── ROI rectangle — orange ───────────────────────────────────────────
        private val roiLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color       = Color.parseColor("#FFFF6D00")
            style       = Paint.Style.STROKE
            strokeWidth = 3f
        }

        // ── Pool table rectangle — blue ──────────────────────────────────────
        private val poolLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color       = Color.parseColor("#FF0099FF")
            style       = Paint.Style.STROKE
            strokeWidth = 3f
        }

        // ── Pocket circles on pool table ─────────────────────────────────────
        private val pocketFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#440099FF")
            style = Paint.Style.FILL
        }

        private val pocketOutlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color       = Color.parseColor("#FF0099FF")
            style       = Paint.Style.STROKE
            strokeWidth = 3f
        }

        // ── Debug cushion outline / pocket-capture circle overlay ───────────
        private val cushionDebugPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#FFFF00E5")
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }
        private val pocketDebugPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#FFFFE800")
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)

            // No darkening — draw ONLY the coloured lines and circles.

            // ── Draw ROI rectangle (orange) ──────────────────────────────────
            val rx1 = roiX1.toFloat()
            val ry1 = roiY1.toFloat()
            val rx2 = roiX2.toFloat()
            val ry2 = roiY2.toFloat()

            canvas.drawLine(rx1, ry1, rx2, ry1, roiLinePaint)   // top
            canvas.drawLine(rx1, ry2, rx2, ry2, roiLinePaint)   // bottom
            canvas.drawLine(rx1, ry1, rx1, ry2, roiLinePaint)   // left
            canvas.drawLine(rx2, ry1, rx2, ry2, roiLinePaint)   // right

            // ── Draw Pool Table rectangle (blue) ─────────────────────────────
            val px1 = poolX1.toFloat()
            val py1 = poolY1.toFloat()
            val px2 = poolX2.toFloat()
            val py2 = poolY2.toFloat()

            canvas.drawLine(px1, py1, px2, py1, poolLinePaint)  // top
            canvas.drawLine(px1, py2, px2, py2, poolLinePaint)  // bottom
            canvas.drawLine(px1, py1, px1, py2, poolLinePaint)  // left
            canvas.drawLine(px2, py1, px2, py2, poolLinePaint)  // right

            // ── Draw pocket circles on pool table only ───────────────────────
            drawPocketCircles(canvas, px1, py1, px2, py2)

            if (showCushionDebug) {
                drawCushionDebugOverlay(canvas, px1, py1, px2, py2)
            }
        }

        /**
         * Pocket positions are computed relative to the POOL TABLE rectangle,
         * never the ROI rectangle.
         *
         * Corner pockets are drawn at the literal, unmodified rectangle
         * corners — they are never offset in any direction.
         *
         * Only the two middle (rail) pockets are shifted, outward along Y,
         * by [pocketNsShift]:
         *   top-middle    → Y − pocketNsShift  (northward, away from table)
         *   bottom-middle → Y + pocketNsShift  (southward, away from table)
         */
        private fun drawPocketCircles(
            canvas: Canvas,
            px1: Float, py1: Float,
            px2: Float, py2: Float
        ) {
            val tblW   = px2 - px1
            val r      = pocketR.toFloat()
            val shift  = pocketNsShift.toFloat()
            val mx     = tblW * 0.5f             // horizontal midpoint offset from px1

            val pockets = arrayOf(
                // ── Corner pockets — pinned exactly to the rectangle corners ──
                floatArrayOf(px1, py1),                  // top-left     (exact corner)
                floatArrayOf(px2, py1),                  // top-right    (exact corner)
                floatArrayOf(px1, py2),                  // bottom-left  (exact corner)
                floatArrayOf(px2, py2),                  // bottom-right (exact corner)
                // ── Middle rail pockets — outward N/S shift along Y only ──────
                floatArrayOf(px1 + mx,   py1 - shift),    // top-middle   (↑ northward)
                floatArrayOf(px1 + mx,   py2 + shift)     // bottom-middle(↓ southward)
            )

            for ((cx, cy) in pockets) {
                canvas.drawCircle(cx, cy, r, pocketFillPaint)
                canvas.drawCircle(cx, cy, r, pocketOutlinePaint)
            }
        }

        /**
         * Draws the true physical cushion outline (with pocket-mouth notches) and
         * the true pocket-capture circles, computed from the calibrated pool
         * table rectangle. poolX1/Y1/X2/Y2 represent only the plain playing
         * rectangle — this derives the full cushion/pocket geometry from it using
         * the same fixed real-world table proportions used by the shot-prediction
         * physics, purely for visual verification. Does not affect calibration
         * values.
         */
        private fun drawCushionDebugOverlay(
            canvas: Canvas,
            px1: Float, py1: Float,
            px2: Float, py2: Float
        ) {
            val rectW = (px2 - px1).toDouble()
            val rectH = (py2 - py1).toDouble()
            if (rectW < 1.0 || rectH < 1.0) return

            val pxPerCmX = rectW / TABLE_LENGTH_CM
            val pxPerCmY = rectH / TABLE_WIDTH_CM
            val pxPerCmAvg = kotlin.math.sqrt(pxPerCmX * pxPerCmY)
            val centerX = (px1 + px2) / 2.0
            val centerY = (py1 + py2) / 2.0

            fun cmToPx(cm: Pair<Double, Double>): Pair<Float, Float> {
                return Pair(
                    (centerX + cm.first * pxPerCmX).toFloat(),
                    (centerY + cm.second * pxPerCmY).toFloat()
                )
            }

            val polygon = buildCushionPolygonCm()
            if (polygon.isNotEmpty()) {
                val path = android.graphics.Path()
                val first = cmToPx(polygon[0])
                path.moveTo(first.first, first.second)
                for (i in 1 until polygon.size) {
                    val p = cmToPx(polygon[i])
                    path.lineTo(p.first, p.second)
                }
                path.close()
                canvas.drawPath(path, cushionDebugPaint)
            }

            val pocketRadiusPx = (POCKET_RADIUS_CM * pxPerCmAvg).toFloat()
            for (pk in POCKET_CENTERS_CM) {
                val p = cmToPx(pk)
                canvas.drawCircle(p.first, p.second, pocketRadiusPx, pocketDebugPaint)
            }
        }
    }
}