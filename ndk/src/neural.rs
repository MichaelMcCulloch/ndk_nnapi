use ffi::{size_t, ANeuralNetworksDevice};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::{convert::TryFrom, mem::size_of, os::raw::c_void, ptr::NonNull};

#[cfg(feature = "api-level-26")]
use crate::hardware_buffer::HardwareBuffer;

pub trait BufferData: Sized {
    fn as_raw_ptr(&self) -> *const c_void;
    fn byte_size() -> usize {
        size_of::<Self>()
    }
}

impl<T> BufferData for T {
    fn as_raw_ptr(&self) -> *const c_void {
        let ptr: *const T = self;
        ptr as *const c_void
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum OperandCode {
    Float32 = ffi::OperandCode::ANEURALNETWORKS_FLOAT32.0,
    Int32 = ffi::OperandCode::ANEURALNETWORKS_INT32.0,
    Uint32 = ffi::OperandCode::ANEURALNETWORKS_UINT32.0,
    TensorFloat32 = ffi::OperandCode::ANEURALNETWORKS_TENSOR_FLOAT32.0,
    TensorInt32 = ffi::OperandCode::ANEURALNETWORKS_TENSOR_INT32.0,
    TensorQuant8Asymm = ffi::OperandCode::ANEURALNETWORKS_TENSOR_QUANT8_ASYMM.0,
    Bool = ffi::OperandCode::ANEURALNETWORKS_BOOL.0,
    TensorQuant16Symm = ffi::OperandCode::ANEURALNETWORKS_TENSOR_QUANT16_SYMM.0,
    TensorFloat16 = ffi::OperandCode::ANEURALNETWORKS_TENSOR_FLOAT16.0,
    TensorBool8 = ffi::OperandCode::ANEURALNETWORKS_TENSOR_BOOL8.0,
    Float16 = ffi::OperandCode::ANEURALNETWORKS_FLOAT16.0,
    TensorQuant8SymmPerChannel = ffi::OperandCode::ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL.0,
    TensorQuant16Asymm = ffi::OperandCode::ANEURALNETWORKS_TENSOR_QUANT16_ASYMM.0,
    TensorQuant8Symm = ffi::OperandCode::ANEURALNETWORKS_TENSOR_QUANT8_SYMM.0,
    TensorQuant8AsymmSigned = ffi::OperandCode::ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED.0,
    Model = ffi::OperandCode::ANEURALNETWORKS_MODEL.0,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum OperationCode {
    Add = ffi::OperationCode::ANEURALNETWORKS_ADD.0,
    AveragePool2d = ffi::OperationCode::ANEURALNETWORKS_AVERAGE_POOL_2D.0,
    Concatenation = ffi::OperationCode::ANEURALNETWORKS_CONCATENATION.0,
    Conv2d = ffi::OperationCode::ANEURALNETWORKS_CONV_2D.0,
    DepthwiseConv2d = ffi::OperationCode::ANEURALNETWORKS_DEPTHWISE_CONV_2D.0,
    DepthToSpace = ffi::OperationCode::ANEURALNETWORKS_DEPTH_TO_SPACE.0,
    Dequantize = ffi::OperationCode::ANEURALNETWORKS_DEQUANTIZE.0,
    EmbeddingLookup = ffi::OperationCode::ANEURALNETWORKS_EMBEDDING_LOOKUP.0,
    Floor = ffi::OperationCode::ANEURALNETWORKS_FLOOR.0,
    FullyConnected = ffi::OperationCode::ANEURALNETWORKS_FULLY_CONNECTED.0,
    HashtableLookup = ffi::OperationCode::ANEURALNETWORKS_HASHTABLE_LOOKUP.0,
    L2Normalization = ffi::OperationCode::ANEURALNETWORKS_L2_NORMALIZATION.0,
    L2Pool2d = ffi::OperationCode::ANEURALNETWORKS_L2_POOL_2D.0,
    LocalResponseNormalization = ffi::OperationCode::ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION.0,
    LOGISTIC = ffi::OperationCode::ANEURALNETWORKS_LOGISTIC.0,
    LshProjection = ffi::OperationCode::ANEURALNETWORKS_LSH_PROJECTION.0,
    Lstm = ffi::OperationCode::ANEURALNETWORKS_LSTM.0,
    MaxPool2d = ffi::OperationCode::ANEURALNETWORKS_MAX_POOL_2D.0,
    Mul = ffi::OperationCode::ANEURALNETWORKS_MUL.0,
    Relu = ffi::OperationCode::ANEURALNETWORKS_RELU.0,
    Relu1 = ffi::OperationCode::ANEURALNETWORKS_RELU1.0,
    Relu6 = ffi::OperationCode::ANEURALNETWORKS_RELU6.0,
    Reshape = ffi::OperationCode::ANEURALNETWORKS_RESHAPE.0,
    ResizeBilinear = ffi::OperationCode::ANEURALNETWORKS_RESIZE_BILINEAR.0,
    Rnn = ffi::OperationCode::ANEURALNETWORKS_RNN.0,
    Softmax = ffi::OperationCode::ANEURALNETWORKS_SOFTMAX.0,
    SpaceToDepth = ffi::OperationCode::ANEURALNETWORKS_SPACE_TO_DEPTH.0,
    Svdf = ffi::OperationCode::ANEURALNETWORKS_SVDF.0,
    Tanh = ffi::OperationCode::ANEURALNETWORKS_TANH.0,
    BatchToSpaceNd = ffi::OperationCode::ANEURALNETWORKS_BATCH_TO_SPACE_ND.0,
    Div = ffi::OperationCode::ANEURALNETWORKS_DIV.0,
    Mean = ffi::OperationCode::ANEURALNETWORKS_MEAN.0,
    Pad = ffi::OperationCode::ANEURALNETWORKS_PAD.0,
    SpaceToBatchNd = ffi::OperationCode::ANEURALNETWORKS_SPACE_TO_BATCH_ND.0,
    Squeeze = ffi::OperationCode::ANEURALNETWORKS_SQUEEZE.0,
    StridedSlice = ffi::OperationCode::ANEURALNETWORKS_STRIDED_SLICE.0,
    Sub = ffi::OperationCode::ANEURALNETWORKS_SUB.0,
    Transpose = ffi::OperationCode::ANEURALNETWORKS_TRANSPOSE.0,
    Abs = ffi::OperationCode::ANEURALNETWORKS_ABS.0,
    Argmax = ffi::OperationCode::ANEURALNETWORKS_ARGMAX.0,
    Argmin = ffi::OperationCode::ANEURALNETWORKS_ARGMIN.0,
    AxisAlignedBboxTransform = ffi::OperationCode::ANEURALNETWORKS_AXIS_ALIGNED_BBOX_TRANSFORM.0,
    BidirectionalSequenceLstm = ffi::OperationCode::ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_LSTM.0,
    BidirectionalSequenceRnn = ffi::OperationCode::ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_RNN.0,
    BoxWithNmsLimit = ffi::OperationCode::ANEURALNETWORKS_BOX_WITH_NMS_LIMIT.0,
    Cast = ffi::OperationCode::ANEURALNETWORKS_CAST.0,
    ChannelShuffle = ffi::OperationCode::ANEURALNETWORKS_CHANNEL_SHUFFLE.0,
    DetectionPostprocessing = ffi::OperationCode::ANEURALNETWORKS_DETECTION_POSTPROCESSING.0,
    Equal = ffi::OperationCode::ANEURALNETWORKS_EQUAL.0,
    Exp = ffi::OperationCode::ANEURALNETWORKS_EXP.0,
    ExpandDims = ffi::OperationCode::ANEURALNETWORKS_EXPAND_DIMS.0,
    Gather = ffi::OperationCode::ANEURALNETWORKS_GATHER.0,
    GenerateProposals = ffi::OperationCode::ANEURALNETWORKS_GENERATE_PROPOSALS.0,
    Greater = ffi::OperationCode::ANEURALNETWORKS_GREATER.0,
    GreaterEqual = ffi::OperationCode::ANEURALNETWORKS_GREATER_EQUAL.0,
    GroupedConv2d = ffi::OperationCode::ANEURALNETWORKS_GROUPED_CONV_2D.0,
    HeatmapMaxKeypoint = ffi::OperationCode::ANEURALNETWORKS_HEATMAP_MAX_KEYPOINT.0,
    InstanceNormalization = ffi::OperationCode::ANEURALNETWORKS_INSTANCE_NORMALIZATION.0,
    Less = ffi::OperationCode::ANEURALNETWORKS_LESS.0,
    LessEqual = ffi::OperationCode::ANEURALNETWORKS_LESS_EQUAL.0,
    Log = ffi::OperationCode::ANEURALNETWORKS_LOG.0,
    LogicalAnd = ffi::OperationCode::ANEURALNETWORKS_LOGICAL_AND.0,
    LogicalNot = ffi::OperationCode::ANEURALNETWORKS_LOGICAL_NOT.0,
    LogicalOr = ffi::OperationCode::ANEURALNETWORKS_LOGICAL_OR.0,
    LogSoftmax = ffi::OperationCode::ANEURALNETWORKS_LOG_SOFTMAX.0,
    Maximum = ffi::OperationCode::ANEURALNETWORKS_MAXIMUM.0,
    Minimum = ffi::OperationCode::ANEURALNETWORKS_MINIMUM.0,
    Neg = ffi::OperationCode::ANEURALNETWORKS_NEG.0,
    NotEqual = ffi::OperationCode::ANEURALNETWORKS_NOT_EQUAL.0,
    PadV2 = ffi::OperationCode::ANEURALNETWORKS_PAD_V2.0,
    Pow = ffi::OperationCode::ANEURALNETWORKS_POW.0,
    Prelu = ffi::OperationCode::ANEURALNETWORKS_PRELU.0,
    Quantize = ffi::OperationCode::ANEURALNETWORKS_QUANTIZE.0,
    Quantized16bitLstm = ffi::OperationCode::ANEURALNETWORKS_QUANTIZED_16BIT_LSTM.0,
    RandomMultinomial = ffi::OperationCode::ANEURALNETWORKS_RANDOM_MULTINOMIAL.0,
    ReduceAll = ffi::OperationCode::ANEURALNETWORKS_REDUCE_ALL.0,
    ReduceAny = ffi::OperationCode::ANEURALNETWORKS_REDUCE_ANY.0,
    ReduceMax = ffi::OperationCode::ANEURALNETWORKS_REDUCE_MAX.0,
    ReduceMin = ffi::OperationCode::ANEURALNETWORKS_REDUCE_MIN.0,
    ReduceProd = ffi::OperationCode::ANEURALNETWORKS_REDUCE_PROD.0,
    ReduceSum = ffi::OperationCode::ANEURALNETWORKS_REDUCE_SUM.0,
    RoiAlign = ffi::OperationCode::ANEURALNETWORKS_ROI_ALIGN.0,
    RoiPooling = ffi::OperationCode::ANEURALNETWORKS_ROI_POOLING.0,
    Rsqrt = ffi::OperationCode::ANEURALNETWORKS_RSQRT.0,
    Select = ffi::OperationCode::ANEURALNETWORKS_SELECT.0,
    Sin = ffi::OperationCode::ANEURALNETWORKS_SIN.0,
    Slice = ffi::OperationCode::ANEURALNETWORKS_SLICE.0,
    Split = ffi::OperationCode::ANEURALNETWORKS_SPLIT.0,
    Sqrt = ffi::OperationCode::ANEURALNETWORKS_SQRT.0,
    Tile = ffi::OperationCode::ANEURALNETWORKS_TILE.0,
    TopkV2 = ffi::OperationCode::ANEURALNETWORKS_TOPK_V2.0,
    TransposeConv2d = ffi::OperationCode::ANEURALNETWORKS_TRANSPOSE_CONV_2D.0,
    UnidirectionalSequenceLstm = ffi::OperationCode::ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM.0,
    UnidirectionalSequenceRnn = ffi::OperationCode::ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_RNN.0,
    ResizeNearestNeighbor = ffi::OperationCode::ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR.0,
    QuantizedLstm = ffi::OperationCode::ANEURALNETWORKS_QUANTIZED_LSTM.0,
    If = ffi::OperationCode::ANEURALNETWORKS_IF.0,
    While = ffi::OperationCode::ANEURALNETWORKS_WHILE.0,
    Elu = ffi::OperationCode::ANEURALNETWORKS_ELU.0,
    HardSwish = ffi::OperationCode::ANEURALNETWORKS_HARD_SWISH.0,
    Fill = ffi::OperationCode::ANEURALNETWORKS_FILL.0,
    Rank = ffi::OperationCode::ANEURALNETWORKS_RANK.0,
    BatchMatmul = ffi::OperationCode::ANEURALNETWORKS_BATCH_MATMUL.0,
    Pack = ffi::OperationCode::ANEURALNETWORKS_PACK.0,
    MirrorPad = ffi::OperationCode::ANEURALNETWORKS_MIRROR_PAD.0,
    Reverse = ffi::OperationCode::ANEURALNETWORKS_REVERSE.0,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum PreferenceCode {
    LowPower = ffi::PreferenceCode::ANEURALNETWORKS_PREFER_LOW_POWER.0,
    FastSingleAnswer = ffi::PreferenceCode::ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER.0,
    SustainedSpeed = ffi::PreferenceCode::ANEURALNETWORKS_PREFER_SUSTAINED_SPEED.0,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum FuseCode {
    None = ffi::FuseCode::ANEURALNETWORKS_FUSED_NONE.0,
    RELU = ffi::FuseCode::ANEURALNETWORKS_FUSED_RELU.0,
    RELU1 = ffi::FuseCode::ANEURALNETWORKS_FUSED_RELU1.0,
    RELU6 = ffi::FuseCode::ANEURALNETWORKS_FUSED_RELU6.0,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum PaddingCode {
    Same = ffi::PaddingCode::ANEURALNETWORKS_PADDING_SAME.0,
    Valid = ffi::PaddingCode::ANEURALNETWORKS_PADDING_VALID.0,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum DeviceTypeCode {
    Unknown = ffi::DeviceTypeCode::ANEURALNETWORKS_DEVICE_UNKNOWN.0,
    Other = ffi::DeviceTypeCode::ANEURALNETWORKS_DEVICE_OTHER.0,
    Cpu = ffi::DeviceTypeCode::ANEURALNETWORKS_DEVICE_CPU.0,
    Gpu = ffi::DeviceTypeCode::ANEURALNETWORKS_DEVICE_GPU.0,
    Accelerator = ffi::DeviceTypeCode::ANEURALNETWORKS_DEVICE_ACCELERATOR.0,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum FeatureLevelCode {
    Level1 = ffi::FeatureLevelCode::ANEURALNETWORKS_FEATURE_LEVEL_1.0,
    Level2 = ffi::FeatureLevelCode::ANEURALNETWORKS_FEATURE_LEVEL_2.0,
    Level3 = ffi::FeatureLevelCode::ANEURALNETWORKS_FEATURE_LEVEL_3.0,
    Level4 = ffi::FeatureLevelCode::ANEURALNETWORKS_FEATURE_LEVEL_4.0,
    Level5 = ffi::FeatureLevelCode::ANEURALNETWORKS_FEATURE_LEVEL_5.0,
    Level6 = ffi::FeatureLevelCode::ANEURALNETWORKS_FEATURE_LEVEL_6.0,
    Level7 = ffi::FeatureLevelCode::ANEURALNETWORKS_FEATURE_LEVEL_7.0,
    Level8 = ffi::FeatureLevelCode::ANEURALNETWORKS_FEATURE_LEVEL_8.0,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
enum ResultCodeBind {
    NoError = ffi::ResultCode::ANEURALNETWORKS_NO_ERROR.0,
    OutOfMemory = ffi::ResultCode::ANEURALNETWORKS_OUT_OF_MEMORY.0,
    Incomplete = ffi::ResultCode::ANEURALNETWORKS_INCOMPLETE.0,
    UnexpectedNull = ffi::ResultCode::ANEURALNETWORKS_UNEXPECTED_NULL.0,
    BadData = ffi::ResultCode::ANEURALNETWORKS_BAD_DATA.0,
    OpFailed = ffi::ResultCode::ANEURALNETWORKS_OP_FAILED.0,
    BadState = ffi::ResultCode::ANEURALNETWORKS_BAD_STATE.0,
    Unmappable = ffi::ResultCode::ANEURALNETWORKS_UNMAPPABLE.0,
    OutputInsufficientSize = ffi::ResultCode::ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE.0,
    UnavailableDevice = ffi::ResultCode::ANEURALNETWORKS_UNAVAILABLE_DEVICE.0,
    MissedDeadlineTransient = ffi::ResultCode::ANEURALNETWORKS_MISSED_DEADLINE_TRANSIENT.0,
    MissedDeadlinePersistent = ffi::ResultCode::ANEURALNETWORKS_MISSED_DEADLINE_PERSISTENT.0,
    ResourceExhaustedTransient = ffi::ResultCode::ANEURALNETWORKS_RESOURCE_EXHAUSTED_TRANSIENT.0,
    ResourceExhaustedPersistent = ffi::ResultCode::ANEURALNETWORKS_RESOURCE_EXHAUSTED_PERSISTENT.0,
    DeadObject = ffi::ResultCode::ANEURALNETWORKS_DEAD_OBJECT.0,
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResultCode {
    NoError,
    OutOfMemory,
    Incomplete,
    UnexpectedNull,
    BadData,
    OpFailed,
    BadState,
    Unmappable,
    OutputInsufficientSize,
    UnavailableDevice,
    MissedDeadlineTransient,
    MissedDeadlinePersistent,
    ResourceExhaustedTransient,
    ResourceExhaustedPersistent,
    DeadObject,
    Undocumented(i32),
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum DurationCode {
    DurationOnHardware = ffi::DurationCode::ANEURALNETWORKS_DURATION_ON_HARDWARE.0,
    DurationInDriver = ffi::DurationCode::ANEURALNETWORKS_DURATION_IN_DRIVER.0,
    FencedDurationOnHardware = ffi::DurationCode::ANEURALNETWORKS_FENCED_DURATION_ON_HARDWARE.0,
    FencedDurationInDriver = ffi::DurationCode::ANEURALNETWORKS_FENCED_DURATION_IN_DRIVER.0,
}
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
pub enum PriorityCode {
    Low = ffi::PriorityCode::ANEURALNETWORKS_PRIORITY_LOW.0,
    Medium = ffi::PriorityCode::ANEURALNETWORKS_PRIORITY_MEDIUM.0,
    High = ffi::PriorityCode::ANEURALNETWORKS_PRIORITY_HIGH.0,
}
impl Default for PriorityCode {
    fn default() -> Self {
        Self::Medium
    }
}

impl Into<ResultCode> for i32 {
    fn into(self) -> ResultCode {
        match ResultCodeBind::try_from(self as u32) {
            Ok(e) => e.into(),
            Err(_) => ResultCode::Undocumented(self),
        }
    }
}

impl Into<ResultCode> for ResultCodeBind {
    fn into(self) -> ResultCode {
        match self {
            ResultCodeBind::NoError => ResultCode::NoError,
            ResultCodeBind::OutOfMemory => ResultCode::OutOfMemory,
            ResultCodeBind::Incomplete => ResultCode::Incomplete,
            ResultCodeBind::UnexpectedNull => ResultCode::UnexpectedNull,
            ResultCodeBind::BadData => ResultCode::BadData,
            ResultCodeBind::OpFailed => ResultCode::OpFailed,
            ResultCodeBind::BadState => ResultCode::BadState,
            ResultCodeBind::Unmappable => ResultCode::Unmappable,
            ResultCodeBind::OutputInsufficientSize => ResultCode::OutputInsufficientSize,
            ResultCodeBind::UnavailableDevice => ResultCode::UnavailableDevice,
            ResultCodeBind::MissedDeadlineTransient => ResultCode::MissedDeadlineTransient,
            ResultCodeBind::MissedDeadlinePersistent => ResultCode::MissedDeadlinePersistent,
            ResultCodeBind::ResourceExhaustedTransient => ResultCode::ResourceExhaustedTransient,
            ResultCodeBind::ResourceExhaustedPersistent => ResultCode::ResourceExhaustedPersistent,
            ResultCodeBind::DeadObject => ResultCode::DeadObject,
        }
    }
}

#[derive(Debug)]
pub struct NeuralNetworksCompilation {
    inner: NonNull<ffi::ANeuralNetworksCompilation>,
}

impl NeuralNetworksCompilation {
    pub fn new(model: &NeuralNetworksModel) -> Result<Self, ResultCode> {
        let mut compilation_ptr = std::ptr::null_mut();
        let result = unsafe {
            ffi::ANeuralNetworksCompilation_create(model.inner.as_ptr(), &mut compilation_ptr)
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(Self {
                inner: NonNull::new(compilation_ptr).unwrap(),
            })
        }
    }

    pub fn set_preference(&mut self, preference: PreferenceCode) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksCompilation_setPreference(self.inner.as_ptr(), preference as i32)
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn finish(&mut self) -> Result<(), ResultCode> {
        let result = unsafe { ffi::ANeuralNetworksCompilation_finish(self.inner.as_ptr()) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn set_priority(&mut self, priority: PriorityCode) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksCompilation_setPriority(self.inner.as_ptr(), priority as i32)
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn set_timeout(&mut self, duration: u64) -> Result<(), ResultCode> {
        let result =
            unsafe { ffi::ANeuralNetworksCompilation_setTimeout(self.inner.as_ptr(), duration) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn get_preferred_memory_alignment_for_input(&self, index: u32) -> Result<u32, ResultCode> {
        let mut alignment = 0;
        let result = unsafe {
            ffi::ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput(
                self.inner.as_ptr(),
                index,
                &mut alignment,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(alignment)
        }
    }

    pub fn get_preferred_memory_padding_for_input(&self, index: u32) -> Result<u32, ResultCode> {
        let mut padding = 0;
        let result = unsafe {
            ffi::ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput(
                self.inner.as_ptr(),
                index,
                &mut padding,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(padding)
        }
    }

    pub fn get_preferred_memory_alignment_for_output(&self, index: u32) -> Result<u32, ResultCode> {
        let mut alignment = 0;
        let result = unsafe {
            ffi::ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput(
                self.inner.as_ptr(),
                index,
                &mut alignment,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(alignment)
        }
    }

    pub fn get_preferred_memory_padding_for_output(&self, index: u32) -> Result<u32, ResultCode> {
        let mut padding = 0;
        let result = unsafe {
            ffi::ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput(
                self.inner.as_ptr(),
                index,
                &mut padding,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(padding)
        }
    }

    pub fn create_for_devices(
        model: &NeuralNetworksModel,
        devices: &[&NeuralNetworksDevice],
    ) -> Result<Self, ResultCode> {
        let mut compilation_ptr = std::ptr::null_mut();
        let devices_ptr: Vec<*const ffi::ANeuralNetworksDevice> = devices
            .iter()
            .map(|device| {
                let d: *const _ = device.inner.as_ptr();
                d
            })
            .collect();
        let result = unsafe {
            ffi::ANeuralNetworksCompilation_createForDevices(
                model.inner.as_ptr(),
                devices_ptr.as_ptr(),
                devices.len() as u32,
                &mut compilation_ptr,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(Self {
                inner: NonNull::new(compilation_ptr).unwrap(),
            })
        }
    }

    pub fn set_caching(&mut self, cache_dir: &str, token: &[u8; 32]) -> Result<(), ResultCode> {
        let cache_dir_cstr = std::ffi::CString::new(cache_dir).unwrap();
        let result = unsafe {
            ffi::ANeuralNetworksCompilation_setCaching(
                self.inner.as_ptr(),
                cache_dir_cstr.as_ptr(),
                token.as_ptr(),
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
}

impl Drop for NeuralNetworksCompilation {
    fn drop(&mut self) {
        unsafe { ffi::ANeuralNetworksCompilation_free(self.inner.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct NeuralNetworksBurst {
    inner: NonNull<ffi::ANeuralNetworksBurst>,
}

impl NeuralNetworksBurst {
    pub fn new(compilation: &NeuralNetworksCompilation) -> Result<Self, ResultCode> {
        let mut burst_ptr = std::ptr::null_mut();
        let result =
            unsafe { ffi::ANeuralNetworksBurst_create(compilation.inner.as_ptr(), &mut burst_ptr) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(Self {
                inner: NonNull::new(burst_ptr).unwrap(),
            })
        }
    }
}

impl Drop for NeuralNetworksBurst {
    fn drop(&mut self) {
        unsafe { ffi::ANeuralNetworksBurst_free(self.inner.as_ptr()) };
    }
}

#[derive(Debug, Clone)]
pub struct NeuralNetworksOperandType {
    inner: NonNull<ffi::ANeuralNetworksOperandType>,
}

impl NeuralNetworksOperandType {
    pub fn new(
        type_: OperandCode,
        dimension_count: u32,
        dimensions: &[u32],
        scale: f32,
        zero_point: i32,
    ) -> Self {
        let mut result = ffi::ANeuralNetworksOperandType {
            type_: type_ as i32,
            dimensionCount: dimension_count,
            dimensions: dimensions.as_ptr(),
            scale,
            zeroPoint: zero_point,
        };

        Self {
            inner: NonNull::new(&mut result).unwrap(),
        }
    }
}

impl Drop for NeuralNetworksOperandType {
    fn drop(&mut self) {
        drop(self.inner);
    }
}

#[derive(Debug)]
pub struct NeuralNetworksDevice {
    inner: NonNull<ffi::ANeuralNetworksDevice>,
}

impl NeuralNetworksDevice {
    pub fn get_name(&self) -> Result<&str, ResultCode> {
        let mut name_ptr = std::ptr::null();
        let result =
            unsafe { ffi::ANeuralNetworksDevice_getName(self.inner.as_ptr(), &mut name_ptr) };

        if result != 0 {
            Err(result.into())
        } else {
            let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
            Ok(name.to_str().unwrap())
        }
    }

    pub fn get_type(&self) -> Result<DeviceTypeCode, ResultCode> {
        let mut type_code = 0;
        let result =
            unsafe { ffi::ANeuralNetworksDevice_getType(self.inner.as_ptr(), &mut type_code) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(DeviceTypeCode::try_from(type_code as u32).unwrap())
        }
    }

    pub fn get_version(&self) -> Result<&str, ResultCode> {
        let mut version_ptr = std::ptr::null();
        let result =
            unsafe { ffi::ANeuralNetworksDevice_getVersion(self.inner.as_ptr(), &mut version_ptr) };

        if result != 0 {
            Err(result.into())
        } else {
            let version = unsafe { std::ffi::CStr::from_ptr(version_ptr) };
            Ok(version.to_str().unwrap())
        }
    }

    pub fn get_feature_level(&self) -> Result<FeatureLevelCode, ResultCode> {
        let mut feature_level = 0;
        let result = unsafe {
            ffi::ANeuralNetworksDevice_getFeatureLevel(self.inner.as_ptr(), &mut feature_level)
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(FeatureLevelCode::try_from(feature_level as u32).unwrap())
        }
    }

    pub fn wait(&self) -> Result<(), ResultCode> {
        let result = unsafe { ffi::ANeuralNetworksDevice_wait(self.inner.as_ptr()) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
}
#[derive(Debug)]
pub struct NeuralNetworksModel {
    inner: NonNull<ffi::ANeuralNetworksModel>,
}

impl NeuralNetworksModel {
    pub fn create() -> Result<Self, ResultCode> {
        let mut model_ptr = std::ptr::null_mut();
        let result = unsafe { ffi::ANeuralNetworksModel_create(&mut model_ptr) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(Self {
                inner: NonNull::new(model_ptr).unwrap(),
            })
        }
    }

    pub fn finish(&mut self) -> Result<(), ResultCode> {
        let result = unsafe { ffi::ANeuralNetworksModel_finish(self.inner.as_ptr()) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn add_operand(
        &mut self,
        operand_type: &NeuralNetworksOperandType,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksModel_addOperand(self.inner.as_ptr(), operand_type.inner.as_ptr())
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn set_operand_value<T: BufferData>(
        &mut self,
        index: i32,
        data: &T,
    ) -> Result<(), ResultCode> {
        let buffer: *const c_void = data.as_raw_ptr();
        let length: usize = T::byte_size();

        let result = unsafe {
            ffi::ANeuralNetworksModel_setOperandValue(
                self.inner.as_ptr(),
                index,
                buffer,
                length as size_t,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn set_operand_symm_per_channel_quant_params(
        &mut self,
        index: i32,
        channel_quant: &ffi::ANeuralNetworksSymmPerChannelQuantParams,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
                self.inner.as_ptr(),
                index,
                channel_quant,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn set_operand_value_from_memory(
        &mut self,
        index: i32,
        memory: &NeuralNetworksMemory,
        offset: usize,
        length: usize,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksModel_setOperandValueFromMemory(
                self.inner.as_ptr(),
                index,
                memory.inner.as_ptr(),
                offset as size_t,
                length as size_t,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn set_operand_value_from_model(
        &mut self,
        index: i32,
        value: &NeuralNetworksModel,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksModel_setOperandValueFromModel(
                self.inner.as_ptr(),
                index,
                value.inner.as_ptr(),
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn add_operation(
        &mut self,
        operation_type: OperationCode,
        input_count: u32,
        inputs: &[u32],
        output_count: u32,
        outputs: &[u32],
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksModel_addOperation(
                self.inner.as_ptr(),
                operation_type as i32,
                input_count,
                inputs.as_ptr(),
                output_count,
                outputs.as_ptr(),
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn identify_inputs_and_outputs(
        &mut self,
        input_count: u32,
        inputs: &[u32],
        output_count: u32,
        outputs: &[u32],
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksModel_identifyInputsAndOutputs(
                self.inner.as_ptr(),
                input_count,
                inputs.as_ptr(),
                output_count,
                outputs.as_ptr(),
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn relax_computation_float32_to_float16(&mut self, allow: bool) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksModel_relaxComputationFloat32toFloat16(self.inner.as_ptr(), allow)
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn get_supported_operations_for_devices(
        &self,
        devices: &[&NeuralNetworksDevice],
        supported_ops: &mut [bool],
    ) -> Result<(), ResultCode> {
        let device_ptrs: Vec<*const ANeuralNetworksDevice> = devices
            .iter()
            .map(|d| {
                let p: *const _ = d.inner.as_ptr();
                p
            })
            .collect();
        let result = unsafe {
            ffi::ANeuralNetworksModel_getSupportedOperationsForDevices(
                self.inner.as_ptr(),
                device_ptrs.as_ptr(),
                devices.len() as u32,
                supported_ops.as_mut_ptr(),
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
}

impl Drop for NeuralNetworksModel {
    fn drop(&mut self) {
        unsafe { ffi::ANeuralNetworksModel_free(self.inner.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct NeuralNetworksExecution {
    inner: NonNull<ffi::ANeuralNetworksExecution>,
}

impl NeuralNetworksExecution {
    pub fn new(compilation: &NeuralNetworksCompilation) -> Result<Self, ResultCode> {
        let mut execution_ptr = std::ptr::null_mut();
        let result = unsafe {
            ffi::ANeuralNetworksExecution_create(compilation.inner.as_ptr(), &mut execution_ptr)
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(Self {
                inner: NonNull::new(execution_ptr).unwrap(),
            })
        }
    }

    pub fn set_input(
        &mut self,
        index: i32,
        operand_type: &NeuralNetworksOperandType,
        buffer: *const std::os::raw::c_void,
        length: usize,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksExecution_setInput(
                self.inner.as_ptr(),
                index,
                operand_type.inner.as_ptr(),
                buffer,
                length as size_t,
            )
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn set_input_from_memory(
        &mut self,
        index: i32,
        operand_type: &NeuralNetworksOperandType,
        memory: &NeuralNetworksMemory,
        offset: usize,
        length: usize,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksExecution_setInputFromMemory(
                self.inner.as_ptr(),
                index,
                operand_type.inner.as_ptr(),
                memory.inner.as_ptr(),
                offset as size_t,
                length as size_t,
            )
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn set_output(
        &mut self,
        index: i32,
        operand_type: &NeuralNetworksOperandType,
        buffer: *mut std::os::raw::c_void,
        length: usize,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksExecution_setOutput(
                self.inner.as_ptr(),
                index,
                operand_type.inner.as_ptr(),
                buffer,
                length as size_t,
            )
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn set_output_from_memory(
        &mut self,
        index: i32,
        operand_type: &NeuralNetworksOperandType,
        memory: &NeuralNetworksMemory,
        offset: usize,
        length: usize,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksExecution_setOutputFromMemory(
                self.inner.as_ptr(),
                index,
                operand_type.inner.as_ptr(),
                memory.inner.as_ptr(),
                offset as size_t,
                length as size_t,
            )
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn start_compute(&mut self) -> Result<NeuralNetworksEvent, ResultCode> {
        let mut event_ptr = std::ptr::null_mut();
        let result = unsafe {
            ffi::ANeuralNetworksExecution_startCompute(self.inner.as_ptr(), &mut event_ptr)
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(NeuralNetworksEvent {
                inner: NonNull::new(event_ptr).unwrap(),
            })
        }
    }
    pub fn set_timeout(&mut self, duration: u64) -> Result<(), ResultCode> {
        let result =
            unsafe { ffi::ANeuralNetworksExecution_setTimeout(self.inner.as_ptr(), duration) };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn set_loop_timeout(&mut self, duration: u64) -> Result<(), ResultCode> {
        let result =
            unsafe { ffi::ANeuralNetworksExecution_setLoopTimeout(self.inner.as_ptr(), duration) };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn start_compute_with_dependencies(
        &mut self,
        dependencies: &[&NeuralNetworksEvent],
        duration: u64,
    ) -> Result<NeuralNetworksEvent, ResultCode> {
        let mut event_ptr = std::ptr::null_mut();
        let dependencies_ptrs: Vec<*const ffi::ANeuralNetworksEvent> = dependencies
            .iter()
            .map(|event| {
                let e: *const _ = event.inner.as_ptr();
                e
            })
            .collect();
        let result = unsafe {
            ffi::ANeuralNetworksExecution_startComputeWithDependencies(
                self.inner.as_ptr(),
                dependencies_ptrs.as_ptr(),
                dependencies.len() as u32,
                duration,
                &mut event_ptr,
            )
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(NeuralNetworksEvent {
                inner: NonNull::new(event_ptr).unwrap(),
            })
        }
    }
    pub fn enable_input_and_output_padding(&mut self, enable: bool) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksExecution_enableInputAndOutputPadding(self.inner.as_ptr(), enable)
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn set_reusable(&mut self, reusable: bool) -> Result<(), ResultCode> {
        let result =
            unsafe { ffi::ANeuralNetworksExecution_setReusable(self.inner.as_ptr(), reusable) };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn compute(&mut self) -> Result<(), ResultCode> {
        let result = unsafe { ffi::ANeuralNetworksExecution_compute(self.inner.as_ptr()) };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn get_output_operand_rank(&mut self, index: i32) -> Result<u32, ResultCode> {
        let mut rank = 0;
        let result = unsafe {
            ffi::ANeuralNetworksExecution_getOutputOperandRank(
                self.inner.as_ptr(),
                index,
                &mut rank,
            )
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(rank)
        }
    }
    pub fn get_output_operand_dimensions(&mut self, index: i32) -> Result<Vec<u32>, ResultCode> {
        let rank = self.get_output_operand_rank(index)?;
        let mut dimensions = vec![0; rank as usize];
        let result = unsafe {
            ffi::ANeuralNetworksExecution_getOutputOperandDimensions(
                self.inner.as_ptr(),
                index,
                dimensions.as_mut_ptr(),
            )
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(dimensions)
        }
    }
    pub fn burst_compute(&mut self, burst: &mut NeuralNetworksBurst) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksExecution_burstCompute(self.inner.as_ptr(), burst.inner.as_ptr())
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn set_measure_timing(&mut self, measure: bool) -> Result<(), ResultCode> {
        let result =
            unsafe { ffi::ANeuralNetworksExecution_setMeasureTiming(self.inner.as_ptr(), measure) };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    pub fn get_duration(&self, duration_code: i32) -> Result<u64, ResultCode> {
        let mut duration = 0;
        let result = unsafe {
            ffi::ANeuralNetworksExecution_getDuration(
                self.inner.as_ptr(),
                duration_code,
                &mut duration,
            )
        };
        if result != 0 {
            Err(result.into())
        } else {
            Ok(duration)
        }
    }
}

impl Drop for NeuralNetworksExecution {
    fn drop(&mut self) {
        unsafe { ffi::ANeuralNetworksExecution_free(self.inner.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct NeuralNetworksEvent {
    inner: NonNull<ffi::ANeuralNetworksEvent>,
}

impl NeuralNetworksEvent {
    pub fn wait(&self) -> Result<(), ResultCode> {
        let result = unsafe { ffi::ANeuralNetworksEvent_wait(self.inner.as_ptr()) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn create_from_sync_fence_fd(sync_fence_fd: i32) -> Result<Self, ResultCode> {
        let mut event = std::ptr::null_mut();
        let result =
            unsafe { ffi::ANeuralNetworksEvent_createFromSyncFenceFd(sync_fence_fd, &mut event) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(Self {
                inner: NonNull::new(event).unwrap(),
            })
        }
    }

    pub fn get_sync_fence_fd(&self) -> Result<i32, ResultCode> {
        let mut sync_fence_fd = 0;
        let result = unsafe {
            ffi::ANeuralNetworksEvent_getSyncFenceFd(self.inner.as_ptr(), &mut sync_fence_fd)
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(sync_fence_fd)
        }
    }
}

impl Drop for NeuralNetworksEvent {
    fn drop(&mut self) {
        unsafe { ffi::ANeuralNetworksEvent_free(self.inner.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct NeuralNetworksMemory {
    inner: NonNull<ffi::ANeuralNetworksMemory>,
}

impl NeuralNetworksMemory {
    pub fn create_from_desc(desc: &NeuralNetworksMemoryDesc) -> Result<Self, ResultCode> {
        let mut memory_ptr = std::ptr::null_mut();
        let result = unsafe {
            ffi::ANeuralNetworksMemory_createFromDesc(desc.inner.as_ptr(), &mut memory_ptr)
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(Self {
                inner: NonNull::new(memory_ptr).unwrap(),
            })
        }
    }

    pub fn copy(src: &NeuralNetworksMemory, dst: &NeuralNetworksMemory) -> Result<(), ResultCode> {
        let result =
            unsafe { ffi::ANeuralNetworksMemory_copy(src.inner.as_ptr(), dst.inner.as_ptr()) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
    #[cfg(feature = "api-level-26")]
    pub fn create_from_hardware_buffer(
        hardware_buffer: &HardwareBuffer,
    ) -> Result<Self, ResultCode> {
        let mut memory_ptr = std::ptr::null_mut();
        let result = unsafe {
            ffi::ANeuralNetworksMemory_createFromAHardwareBuffer(
                hardware_buffer.as_ptr(),
                &mut memory_ptr,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(Self {
                inner: NonNull::new(memory_ptr).unwrap(),
            })
        }
    }
}

impl Drop for NeuralNetworksMemory {
    fn drop(&mut self) {
        unsafe { ffi::ANeuralNetworksMemory_free(self.inner.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct NeuralNetworksMemoryDesc {
    inner: NonNull<ffi::ANeuralNetworksMemoryDesc>,
}

impl NeuralNetworksMemoryDesc {
    pub fn new() -> Result<Self, ResultCode> {
        let mut desc_ptr = std::ptr::null_mut();
        let result = unsafe { ffi::ANeuralNetworksMemoryDesc_create(&mut desc_ptr) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(Self {
                inner: NonNull::new(desc_ptr).unwrap(),
            })
        }
    }

    pub fn add_input_role(
        &mut self,
        compilation: &NeuralNetworksCompilation,
        index: u32,
        frequency: f32,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksMemoryDesc_addInputRole(
                self.inner.as_ptr(),
                compilation.inner.as_ptr(),
                index,
                frequency,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn add_output_role(
        &mut self,
        compilation: &NeuralNetworksCompilation,
        index: u32,
        frequency: f32,
    ) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksMemoryDesc_addOutputRole(
                self.inner.as_ptr(),
                compilation.inner.as_ptr(),
                index,
                frequency,
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn set_dimensions(&mut self, rank: u32, dimensions: &[u32]) -> Result<(), ResultCode> {
        let result = unsafe {
            ffi::ANeuralNetworksMemoryDesc_setDimensions(
                self.inner.as_ptr(),
                rank,
                dimensions.as_ptr(),
            )
        };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }

    pub fn finish(&mut self) -> Result<(), ResultCode> {
        let result = unsafe { ffi::ANeuralNetworksMemoryDesc_finish(self.inner.as_ptr()) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(())
        }
    }
}

impl Drop for NeuralNetworksMemoryDesc {
    fn drop(&mut self) {
        unsafe { ffi::ANeuralNetworksMemoryDesc_free(self.inner.as_ptr()) };
    }
}

#[derive(Debug, Clone)]
pub struct NeuralNetworksSymmPerChannelQuantParams {
    inner: NonNull<ffi::ANeuralNetworksSymmPerChannelQuantParams>,
}

impl NeuralNetworksSymmPerChannelQuantParams {
    pub fn new(channel_dim: u32, scales: &[f32]) -> Self {
        let mut result = ffi::ANeuralNetworksSymmPerChannelQuantParams {
            channelDim: channel_dim,
            scaleCount: scales.len() as u32,
            scales: scales.as_ptr(),
        };

        Self {
            inner: NonNull::new(&mut result).unwrap(),
        }
    }
}

impl Drop for NeuralNetworksSymmPerChannelQuantParams {
    fn drop(&mut self) {
        drop(self.inner)
    }
}
#[derive(Debug)]
pub struct NeuralNetworks;

impl NeuralNetworks {
    pub fn get_default_loop_timeout() -> u64 {
        unsafe { ffi::ANeuralNetworks_getDefaultLoopTimeout() }
    }

    pub fn get_maximum_loop_timeout() -> u64 {
        unsafe { ffi::ANeuralNetworks_getMaximumLoopTimeout() }
    }

    pub fn get_device_count() -> Result<u32, ResultCode> {
        let mut num_devices = 0;
        let result = unsafe { ffi::ANeuralNetworks_getDeviceCount(&mut num_devices) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(num_devices)
        }
    }

    pub fn get_device(dev_index: u32) -> Result<NeuralNetworksDevice, ResultCode> {
        let mut device_ptr = std::ptr::null_mut();
        let result = unsafe { ffi::ANeuralNetworks_getDevice(dev_index, &mut device_ptr) };

        if result != 0 {
            Err(result.into())
        } else {
            Ok(NeuralNetworksDevice {
                inner: NonNull::new(device_ptr).unwrap(),
            })
        }
    }
}
