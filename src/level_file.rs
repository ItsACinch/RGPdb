/// Level file format for binary serialization/deserialization
/// Uses memmap2 for efficient memory-mapped I/O
/// 
/// File Format:
/// [Header: 64 bytes]
/// [NodeSection: node_props array]
/// [EdgeSection: CSR structure (row_ptr, col_idx, edge_props)]
/// [RoomSection: room metadata]
/// [PVSSection: PVS visibility data]

use crate::graph::{Graph, NodeProps, EdgeProps};
use crate::rooms::RoomCollection;
use crate::pvs::PVS;
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use thiserror::Error;

/// Magic number for RGDB level files
const MAGIC: &[u8; 4] = b"RGDB";
const VERSION: u32 = 1;
const HEADER_SIZE: u64 = 64;

/// Maximum allowed file size (prevents DoS)
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10GB

/// Maximum allowed nodes in a file (prevents DoS)
const MAX_NODES: u32 = 100_000_000;
/// Maximum allowed edges in a file (prevents DoS)
const MAX_EDGES: u32 = 1_000_000_000;
/// Maximum allowed rooms in a file (prevents DoS)
const MAX_ROOMS: u32 = 10_000_000;

/// Errors for level file operations
#[derive(Debug, Error)]
pub enum LevelFileError {
    #[error("Invalid magic number")]
    InvalidMagic,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Graph structure mismatch: {0}")]
    StructureMismatch(String),
}

/// Level file header (64 bytes, 8-byte aligned)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct LevelHeader {
    magic: [u8; 4],
    version: u32,
    num_nodes: u32,
    num_edges: u32,
    num_rooms: u32,
    num_angle_bins: u8,
    _padding: [u8; 3], // Align to 8 bytes
    // Offsets to sections (in bytes from start of file)
    node_section_offset: u64,
    edge_section_offset: u64,
    rooms_section_offset: u64,
    pvs_section_offset: u64,
    checksum: u32, // Simple checksum for validation
    _reserved: [u8; 4],
}

impl LevelHeader {
    fn new(
        num_nodes: u32,
        num_edges: u32,
        num_rooms: u32,
        num_angle_bins: u8,
        node_offset: u64,
        edge_offset: u64,
        rooms_offset: u64,
        pvs_offset: u64,
    ) -> Self {
        Self {
            magic: *MAGIC,
            version: VERSION,
            num_nodes,
            num_edges,
            num_rooms,
            num_angle_bins,
            _padding: [0; 3],
            node_section_offset: node_offset,
            edge_section_offset: edge_offset,
            rooms_section_offset: rooms_offset,
            pvs_section_offset: pvs_offset,
            checksum: 0, // Will be computed
            _reserved: [0; 4],
        }
    }
    
    fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.magic)?;
        writer.write_u32::<LittleEndian>(self.version)?;
        writer.write_u32::<LittleEndian>(self.num_nodes)?;
        writer.write_u32::<LittleEndian>(self.num_edges)?;
        writer.write_u32::<LittleEndian>(self.num_rooms)?;
        writer.write_u8(self.num_angle_bins)?;
        writer.write_all(&self._padding)?;
        writer.write_u64::<LittleEndian>(self.node_section_offset)?;
        writer.write_u64::<LittleEndian>(self.edge_section_offset)?;
        writer.write_u64::<LittleEndian>(self.rooms_section_offset)?;
        writer.write_u64::<LittleEndian>(self.pvs_section_offset)?;
        writer.write_u32::<LittleEndian>(self.checksum)?;
        writer.write_all(&self._reserved)?;
        Ok(())
    }
    
    fn read<R: Read>(reader: &mut R) -> Result<Self, LevelFileError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != *MAGIC {
            return Err(LevelFileError::InvalidMagic);
        }
        
        let version = reader.read_u32::<LittleEndian>()?;
        if version != VERSION {
            return Err(LevelFileError::UnsupportedVersion(version));
        }
        
        let num_nodes = reader.read_u32::<LittleEndian>()?;
        if num_nodes > MAX_NODES {
            return Err(LevelFileError::StructureMismatch(format!(
                "Invalid num_nodes: {} (max: {})", num_nodes, MAX_NODES
            )));
        }
        
        let num_edges = reader.read_u32::<LittleEndian>()?;
        if num_edges > MAX_EDGES {
            return Err(LevelFileError::StructureMismatch(format!(
                "Invalid num_edges: {} (max: {})", num_edges, MAX_EDGES
            )));
        }
        
        let num_rooms = reader.read_u32::<LittleEndian>()?;
        if num_rooms > MAX_ROOMS {
            return Err(LevelFileError::StructureMismatch(format!(
                "Invalid num_rooms: {} (max: {})", num_rooms, MAX_ROOMS
            )));
        }
        
        let num_angle_bins = reader.read_u8()?;
        if num_angle_bins == 0 || num_angle_bins > 255 {
            return Err(LevelFileError::StructureMismatch(format!(
                "Invalid num_angle_bins: {}", num_angle_bins
            )));
        }
        let mut padding = [0u8; 3];
        reader.read_exact(&mut padding)?;
        let node_section_offset = reader.read_u64::<LittleEndian>()?;
        let edge_section_offset = reader.read_u64::<LittleEndian>()?;
        let rooms_section_offset = reader.read_u64::<LittleEndian>()?;
        let pvs_section_offset = reader.read_u64::<LittleEndian>()?;
        let checksum = reader.read_u32::<LittleEndian>()?;
        let mut reserved = [0u8; 4];
        reader.read_exact(&mut reserved)?;
        
        Ok(Self {
            magic,
            version,
            num_nodes,
            num_edges,
            num_rooms,
            num_angle_bins,
            _padding: padding,
            node_section_offset,
            edge_section_offset,
            rooms_section_offset,
            pvs_section_offset,
            checksum,
            _reserved: reserved,
        })
    }
}

/// Safe serialization of NodeProps
fn write_node_props<W: Write>(writer: &mut W, props: &NodeProps) -> std::io::Result<()> {
    writer.write_all(&props.luminance.to_le_bytes())?;
    writer.write_all(&props.reflection.to_le_bytes())?;
    writer.write_all(&props.refraction_index.to_le_bytes())?;
    writer.write_u8(props.default_angle_bin)?;
    // Padding to align to 16 bytes
    writer.write_all(&[0u8; 3])?;
    Ok(())
}

/// Safe deserialization of NodeProps
fn read_node_props<R: Read>(reader: &mut R) -> std::io::Result<NodeProps> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let luminance = f32::from_le_bytes(buf);
    reader.read_exact(&mut buf)?;
    let reflection = f32::from_le_bytes(buf);
    reader.read_exact(&mut buf)?;
    let refraction_index = f32::from_le_bytes(buf);
    let default_angle_bin = reader.read_u8()?;
    let mut padding = [0u8; 3];
    reader.read_exact(&mut padding)?;
    
    // Create uniform luminance for backward compatibility
    use crate::property_map::create_uniform_luminance;
    Ok(NodeProps {
        luminance,
        reflection,
        refraction_index,
        default_angle_bin,
        relationship_property: None,
        directional_luminance: create_uniform_luminance(luminance),
    })
}

/// Safe serialization of EdgeProps
fn write_edge_props<W: Write>(writer: &mut W, props: &EdgeProps) -> std::io::Result<()> {
    writer.write_all(&props.attenuation.to_le_bytes())?;
    writer.write_u8(props.angle_bin)?;
    writer.write_u8(if props.is_portal { 1 } else { 0 })?;
    // Padding
    writer.write_all(&[0u8; 2])?;
    Ok(())
}

/// Safe deserialization of EdgeProps
fn read_edge_props<R: Read>(reader: &mut R) -> std::io::Result<EdgeProps> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let attenuation = f32::from_le_bytes(buf);
    let angle_bin = reader.read_u8()?;
    let is_portal = reader.read_u8()? != 0;
    let mut padding = [0u8; 2];
    reader.read_exact(&mut padding)?;
    
    Ok(EdgeProps {
        attenuation,
        angle_bin,
        is_portal,
    })
}

/// Write graph to level file
pub fn write_level_file(
    graph: &Graph,
    rooms: &RoomCollection,
    _pvs: &PVS,
    path: &str,
) -> Result<(), LevelFileError> {
    let mut file = File::create(path)?;
    
    // Calculate section offsets
    let node_section_offset = HEADER_SIZE;
    let node_section_size = (graph.num_nodes() * 16) as u64; // 16 bytes per NodeProps (aligned)
    let edge_section_offset = node_section_offset + node_section_size;
    
    // Edge section: row_ptr (u32 per entry), col_idx (u32 per entry), edge_props (8 bytes per entry)
    let row_ptr_size = ((graph.num_nodes() + 1) * 4) as u64;
    let col_idx_size = (graph.num_edges() * 4) as u64;
    let edge_props_size = (graph.num_edges() * 8) as u64;
    let edge_section_size = row_ptr_size + col_idx_size + edge_props_size;
    
    let rooms_section_offset = edge_section_offset + edge_section_size;
    // Rooms section: room_count (u32) + per room: id (u32), start (u32), count (u32)
    let rooms_section_size = (4 + rooms.num_rooms() * 12) as u64;
    
    let pvs_section_offset = rooms_section_offset + rooms_section_size;
    // PVS section: simplified for now (just placeholder)
    let _pvs_section_size = 0u64; // TODO: implement full PVS serialization
    
    // Write header
    let header = LevelHeader::new(
        graph.num_nodes() as u32,
        graph.num_edges() as u32,
        rooms.num_rooms() as u32,
        crate::graph::N_ANGLE_BINS as u8,
        node_section_offset,
        edge_section_offset,
        rooms_section_offset,
        pvs_section_offset,
    );
    header.write(&mut file)?;
    
    // Write node properties
    file.seek(SeekFrom::Start(node_section_offset))?;
    for props in graph.node_props() {
        write_node_props(&mut file, props)?;
    }
    
    // Write edge section (CSR)
    file.seek(SeekFrom::Start(edge_section_offset))?;
    // Write row_ptr
    for &ptr in graph.row_ptr() {
        file.write_u32::<LittleEndian>(ptr as u32)?;
    }
    // Write col_idx
    for &node_id in graph.col_idx() {
        file.write_u32::<LittleEndian>(node_id)?;
    }
    // Write edge_props
    for props in graph.edge_props() {
        write_edge_props(&mut file, props)?;
    }
    
    // Write room_map
    file.write_u32::<LittleEndian>(graph.room_map().len() as u32)?;
    for &room_id in graph.room_map() {
        file.write_u32::<LittleEndian>(room_id)?;
    }
    
    // Write rooms section
    file.seek(SeekFrom::Start(rooms_section_offset))?;
    file.write_u32::<LittleEndian>(rooms.num_rooms() as u32)?;
    for room in &rooms.rooms {
        file.write_u32::<LittleEndian>(room.id)?;
        file.write_u32::<LittleEndian>(room.node_start as u32)?;
        file.write_u32::<LittleEndian>(room.node_count as u32)?;
    }
    
    // TODO: Write PVS section
    
    Ok(())
}

/// Read graph from level file using mmap
pub fn read_level_file_mmap(path: &str) -> Result<(Graph, RoomCollection, PVS), LevelFileError> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();
    
    // Validate file size before mapping
    if file_size < HEADER_SIZE {
        return Err(LevelFileError::StructureMismatch(
            "File too small to contain header".to_string(),
        ));
    }
    
    // Prevent DoS from extremely large files
    if file_size > MAX_FILE_SIZE {
        return Err(LevelFileError::StructureMismatch(format!(
            "File size {} exceeds maximum allowed {}", file_size, MAX_FILE_SIZE
        )));
    }
    
    // Safe memory mapping with explicit length
    let mmap = unsafe { 
        MmapOptions::new()
            .len(file_size as usize)
            .map(&file)
            .map_err(|e| LevelFileError::Io(e))?
    };
    
    // Validate mmap length matches file size
    if mmap.len() != file_size as usize {
        return Err(LevelFileError::StructureMismatch(
            "Memory map size mismatch".to_string(),
        ));
    }
    
    // Read header
    let mut header_reader = &mmap[..HEADER_SIZE as usize];
    let header = LevelHeader::read(&mut header_reader)?;
    
    // Validate offsets are within file bounds
    if header.node_section_offset >= file_size
        || header.edge_section_offset >= file_size
        || header.rooms_section_offset >= file_size
        || header.pvs_section_offset >= file_size
    {
        return Err(LevelFileError::StructureMismatch(
            "Section offsets exceed file size".to_string(),
        ));
    }
    
    // Validate offsets are in ascending order
    if !(header.node_section_offset < header.edge_section_offset
        && header.edge_section_offset < header.rooms_section_offset
        && header.rooms_section_offset < header.pvs_section_offset)
    {
        return Err(LevelFileError::StructureMismatch(
            "Section offsets not in ascending order".to_string(),
        ));
    }
    
    // Validate offsets are within mmap bounds
    if header.node_section_offset as usize >= mmap.len()
        || header.edge_section_offset as usize >= mmap.len()
        || header.rooms_section_offset as usize >= mmap.len()
        || header.pvs_section_offset as usize >= mmap.len()
    {
        return Err(LevelFileError::StructureMismatch(
            "Section offsets exceed memory map size".to_string(),
        ));
    }
    
    // Read node properties with bounds checking
    let node_section_start = header.node_section_offset as usize;
    let node_section_end = header.edge_section_offset as usize;
    
    // Validate section bounds
    if node_section_start >= mmap.len() || node_section_end > mmap.len() || node_section_start >= node_section_end {
        return Err(LevelFileError::StructureMismatch(
            "Invalid node section bounds".to_string(),
        ));
    }
    
    let mut node_reader = &mmap[node_section_start..node_section_end];
    let mut node_props = Vec::with_capacity(header.num_nodes as usize);
    for _ in 0..header.num_nodes {
        node_props.push(read_node_props(&mut node_reader)?);
    }
    
    // Read edge section with bounds checking
    let edge_section_start = header.edge_section_offset as usize;
    let rooms_section_start = header.rooms_section_offset as usize;
    
    // Validate section bounds
    if edge_section_start >= mmap.len() || rooms_section_start > mmap.len() || edge_section_start >= rooms_section_start {
        return Err(LevelFileError::StructureMismatch(
            "Invalid edge section bounds".to_string(),
        ));
    }
    
    let mut edge_reader = &mmap[edge_section_start..rooms_section_start];
    
    // Read row_ptr
    let mut row_ptr = Vec::with_capacity((header.num_nodes + 1) as usize);
    for _ in 0..=header.num_nodes {
        row_ptr.push(edge_reader.read_u32::<LittleEndian>()? as usize);
    }
    
    // Read col_idx
    let mut col_idx = Vec::with_capacity(header.num_edges as usize);
    for _ in 0..header.num_edges {
        col_idx.push(edge_reader.read_u32::<LittleEndian>()?);
    }
    
    // Read edge_props
    let mut edge_props = Vec::with_capacity(header.num_edges as usize);
    for _ in 0..header.num_edges {
        edge_props.push(read_edge_props(&mut edge_reader)?);
    }
    
    // Read room_map (stored after edge_props)
    let room_map_len = edge_reader.read_u32::<LittleEndian>()? as usize;
    let mut room_map = Vec::with_capacity(room_map_len);
    for _ in 0..room_map_len {
        room_map.push(edge_reader.read_u32::<LittleEndian>()?);
    }
    
    // Reconstruct graph from CSR data
    let graph = Graph::from_csr(
        header.num_nodes as usize,
        row_ptr,
        col_idx,
        node_props,
        edge_props,
        room_map,
    ).map_err(|e| LevelFileError::StructureMismatch(format!("Graph reconstruction failed: {}", e)))?;
    
    // Read rooms section
    let mut rooms_reader = &mmap[rooms_section_start..];
    let num_rooms = rooms_reader.read_u32::<LittleEndian>()? as usize;
    let mut rooms_vec = Vec::with_capacity(num_rooms);
    for _ in 0..num_rooms {
        let id = rooms_reader.read_u32::<LittleEndian>()?;
        let node_start = rooms_reader.read_u32::<LittleEndian>()? as usize;
        let node_count = rooms_reader.read_u32::<LittleEndian>()? as usize;
        rooms_vec.push(crate::rooms::Room {
            id,
            node_start,
            node_count,
        });
    }
    
    // Build room_index
    let max_room_id = rooms_vec.iter().map(|r| r.id).max().unwrap_or(0) as usize + 1;
    let mut room_index = vec![usize::MAX; max_room_id];
    for (idx, room) in rooms_vec.iter().enumerate() {
        room_index[room.id as usize] = idx;
    }
    
    let rooms = RoomCollection {
        rooms: rooms_vec,
        room_index,
    };
    
    // Create empty PVS (TODO: deserialize PVS)
    let pvs = PVS::new(header.num_rooms as usize, header.num_angle_bins as usize);
    
    Ok((graph, rooms, pvs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::rooms::RoomCollection;
    use crate::pvs::PVS;
    use std::fs;
    
    #[test]
    fn test_write_read_roundtrip() {
        // Create a simple graph
        let mut graph = Graph::new(3);
        graph.set_node_props(0, NodeProps::from_uniform_luminance(1.0)).unwrap();
        // Set room assignments
        graph.set_room(0, 0).unwrap();
        graph.set_room(1, 0).unwrap();
        graph.set_room(2, 1).unwrap();
        
        let rooms = RoomCollection::from_room_map(graph.room_map());
        let pvs = PVS::new(2, 16);
        
        let test_path = "test_level.rgdb";
        
        // Write
        write_level_file(&graph, &rooms, &pvs, test_path).unwrap();
        
        // Read
        let (read_graph, read_rooms, _read_pvs) = read_level_file_mmap(test_path).unwrap();
        
        // Verify
        assert_eq!(read_graph.num_nodes(), graph.num_nodes());
        assert_eq!(read_graph.node_props().len(), graph.node_props().len());
        assert_eq!(read_rooms.num_rooms(), rooms.num_rooms());
        
        // Cleanup
        let _ = fs::remove_file(test_path);
    }
}
